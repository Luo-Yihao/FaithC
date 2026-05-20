"""
FCT Decoder: Faithful Contour Token → Mesh

Reconstructs mesh from FCT representation using edge-based quad extraction.
"""

import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from atom3d.grid import OctreeIndexer, CubeGrid
try:
    from torch_scatter import scatter_max
except ImportError:
    from .torch_scatter_compat import scatter_max


@dataclass
class DecodedMesh:
    """Result of FCT decoding."""
    vertices: torch.Tensor  # [V, 3]
    faces: torch.Tensor     # [F, 3]


class FCTDecoder:
    """
    Faithful Contour Token Decoder.
    
    Reconstructs mesh from FCT representation using edge-based extraction:
    1. Find edges with non-zero flux
    2. For each such edge, form a quad from 4 incident voxel anchors
    3. Triangulate quads
    
    Example:
        decoder = FCTDecoder(resolution=128)
        mesh = decoder.decode(fct_result)
        # mesh.vertices: [V, 3]
        # mesh.faces: [F, 3]
    """
    
    def __init__(
        self,
        resolution: int,
        bounds: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            resolution: Grid resolution (must be power of 2)
            bounds: Grid bounds [2, 3], defaults to [-1, 1]^3
            device: Device for computation
        """
        self.resolution = resolution
        self.device = device
        
        max_level = int(torch.log2(torch.tensor(resolution)).item())
        self.grid = OctreeIndexer(max_level=max_level, bounds=bounds, device=device)
    
    def decode(
        self,
        active_voxel_indices: torch.Tensor,
        anchors: torch.Tensor,
        edge_flux_sign: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        triangulation_mode: str = 'auto',
    ) -> DecodedMesh:
        """
        Decode FCT representation to mesh.
        
        Args:
            active_voxel_indices: [K] linear voxel indices
            anchors: [K, 3] anchor points per voxel
            edge_flux_sign: [K, 12] edge flux signs per voxel
            normals: [K, 3] optional normals for triangulation
            triangulation_mode: Quad-to-triangle method:
                - 'auto': use 'normal_abs' if normals provided, else 'length'
                - 'simple_02': fixed 0-2 diagonal split
                - 'simple_13': fixed 1-3 diagonal split  
                - 'length': choose shorter diagonal
                - 'angle': minimize interior angle deviation
                - 'normal': maximize normal consistency (requires normals)
                - 'normal_abs': like 'normal' but uses abs(dot), direction-agnostic
        
        Returns:
            DecodedMesh with vertices and faces
        """
        K = active_voxel_indices.shape[0]
        device = active_voxel_indices.device
        
        if K == 0:
            return DecodedMesh(
                vertices=torch.empty(0, 3, device=device),
                faces=torch.empty(0, 3, dtype=torch.long, device=device)
            )
        
        # Step 1: Get edge indices for all voxels
        voxel_edge_indices = self.grid.cube_edge_indices(active_voxel_indices)  # [K, 12]
        
        # Flatten and get unique edges
        edges_flat = voxel_edge_indices.reshape(-1)
        flux_flat = edge_flux_sign.reshape(-1).float()
        
        unique_edges, inverse = torch.unique(edges_flat, return_inverse=True)
        E = unique_edges.shape[0]
        
        # Step 2: Aggregate flux per unique edge (max absolute value)
        _, best_idx = scatter_max(flux_flat.abs(), inverse, dim_size=E)
        edge_flux = flux_flat[best_idx]  # [E]
        
        # Find edges with flux
        has_flux = edge_flux != 0.0
        
        # Step 3: Get incident cubes for valid edges
        edge_incident_cubes = self.grid.edge_incident_cubes(unique_edges)  # [E, 4]
        
        # Build mapping from global voxel index to local index using searchsorted
        # This avoids allocating a dense array of size num_cubes (which would be 64GB at 2048^3)
        sorted_active_indices, sort_perm = torch.sort(active_voxel_indices)
        
        # Flatten incident cubes for batch searchsorted
        incident_flat = edge_incident_cubes.reshape(-1)  # [E*4]
        
        # Find positions in sorted active indices
        positions = torch.searchsorted(sorted_active_indices, incident_flat)
        
        # Clamp to valid range to avoid out-of-bounds access
        positions = positions.clamp(0, K - 1)
        
        # Check if the found positions actually match (i.e., the cube is active)
        is_active = sorted_active_indices[positions] == incident_flat
        
        # Map back through sort permutation to get local indices
        local_idx_flat = torch.where(
            is_active,
            sort_perm[positions],
            torch.tensor(-1, dtype=torch.long, device=device)
        )
        local_idx = local_idx_flat.reshape(E, 4)  # [E, 4]
        
        # Valid edges: has flux AND all 4 incident cubes are active
        all_active = (local_idx >= 0).all(dim=-1)  # [E]
        valid_edge_mask = has_flux & all_active
        
        valid_edge_ids = torch.where(valid_edge_mask)[0]
        
        if valid_edge_ids.numel() == 0:
            return DecodedMesh(
                vertices=anchors,
                faces=torch.empty(0, 3, dtype=torch.long, device=device)
            )
        
        # Step 4: Form quads from incident cubes
        quads_local = local_idx[valid_edge_ids]  # [Q, 4]
        quads_flux = edge_flux[valid_edge_ids]    # [Q]
        
        # Orient quads based on flux direction
        # edge_incident_cubes returns CW order (original/backward compatible)
        # Flip for positive flux to align normals correctly
        quads_oriented = torch.where(
            quads_flux[:, None] > 0,  # Negative flux (edge opposes surface) → flip to make normal outward
            quads_local.flip(dims=[1]),
            quads_local                  # Negative flux → keep
        )
        
        # Step 5: Get unique vertices used
        used_verts, new_indices = torch.unique(
            quads_oriented.reshape(-1), 
            return_inverse=True
        )
        
        vertices = anchors[used_verts]  # [V, 3]
        quads_compact = new_indices.view(-1, 4)  # [Q, 4]
        
        # Step 6: Triangulate quads based on mode
        mode = triangulation_mode.lower()
        
        if mode == 'auto':
            if normals is not None:
                faces = self._triangulate_by_angle(vertices, quads_compact, normals[used_verts], use_abs_normal=True)
            else:
                faces = self._triangulate_by_length(vertices, quads_compact)
        elif mode == 'simple_02':
            faces = self._triangulate_simple(quads_compact)
        elif mode == 'simple_13':
            faces = self._triangulate_simple_13(quads_compact)
        elif mode == 'length':
            faces = self._triangulate_by_length(vertices, quads_compact)
        elif mode == 'angle':
            faces = self._triangulate_by_angle(vertices, quads_compact, normals=None)
        elif mode == 'normal':
            if normals is None:
                raise ValueError("triangulation_mode='normal' requires normals to be provided")
            faces = self._triangulate_by_angle(vertices, quads_compact, normals[used_verts])
        elif mode == 'normal_abs':
            if normals is None:
                raise ValueError("triangulation_mode='normal_abs' requires normals to be provided")
            faces = self._triangulate_by_angle(vertices, quads_compact, normals[used_verts], use_abs_normal=True)
        else:
            raise ValueError(f"Unknown triangulation_mode: {mode}. "
                           f"Valid options: 'auto', 'simple_02', 'simple_13', 'length', 'angle', 'normal', 'normal_abs'")
        
        return DecodedMesh(vertices=vertices, faces=faces)
    
    def decode_from_result(self, fct_result) -> DecodedMesh:
        """
        Convenience method to decode from FCTResult dataclass.
        
        Args:
            fct_result: FCTResult from encoder
        
        Returns:
            DecodedMesh
        """
        return self.decode(
            active_voxel_indices=fct_result.active_voxel_indices,
            anchors=fct_result.anchor,
            edge_flux_sign=fct_result.edge_flux_sign,
            normals=fct_result.normal
        )
    
    def _triangulate_simple(self, quads: torch.Tensor) -> torch.Tensor:
        """
        Simple quad triangulation: split along 0-2 diagonal.
        
        Args:
            quads: [Q, 4] quad vertex indices
        
        Returns:
            faces: [2*Q, 3] triangle faces
        """
        # Split [0,1,2,3] into [0,1,2] and [0,2,3]
        tri1 = quads[:, [0, 1, 2]]
        tri2 = quads[:, [0, 2, 3]]
        
        faces = torch.cat([tri1, tri2], dim=0)
        return faces
    
    def _triangulate_simple_13(self, quads: torch.Tensor) -> torch.Tensor:
        """
        Simple quad triangulation: split along 1-3 diagonal.
        
        Args:
            quads: [Q, 4] quad vertex indices
        
        Returns:
            faces: [2*Q, 3] triangle faces
        """
        # Split [0,1,2,3] into [0,1,3] and [1,2,3]
        tri1 = quads[:, [0, 1, 3]]
        tri2 = quads[:, [1, 2, 3]]
        
        faces = torch.cat([tri1, tri2], dim=0)
        return faces
    
    def _triangulate_by_length(
        self, 
        vertices: torch.Tensor, 
        quads: torch.Tensor,
        chunk_size: int = 4096
    ) -> torch.Tensor:
        """
        Triangulate quads by choosing the shorter diagonal.
        
        This produces more equilateral triangles by avoiding long, thin splits.
        
        Args:
            vertices: [V, 3] vertex positions
            quads: [Q, 4] quad vertex indices
            chunk_size: Number of quads to process per chunk (for memory efficiency)
        
        Returns:
            faces: [2*Q, 3] triangle faces
        """
        device = quads.device
        num_quads = quads.shape[0]
        
        # Two triangulation patterns
        QUAD2TRI = torch.tensor([
            [[0, 1, 2], [0, 2, 3]],  # Pattern 0: 0-2 diagonal
            [[0, 1, 3], [1, 2, 3]],  # Pattern 1: 1-3 diagonal
        ], device=device, dtype=torch.long)
        
        all_tri_faces = []
        
        for i in range(0, num_quads, chunk_size):
            quads_chunk = quads[i:i + chunk_size]
            quad_coords = vertices[quads_chunk]  # [chunk, 4, 3]
            
            # Compute diagonal lengths
            v0, v1, v2, v3 = quad_coords[:, 0], quad_coords[:, 1], quad_coords[:, 2], quad_coords[:, 3]
            diag_02_len = (v2 - v0).norm(dim=-1)  # [chunk]
            diag_13_len = (v3 - v1).norm(dim=-1)  # [chunk]
            
            # Use 0-2 diagonal if it's shorter, else 1-3
            condition = diag_02_len <= diag_13_len  # [chunk]
            
            # Apply chosen pattern
            faces_pattern0 = quads_chunk[:, QUAD2TRI[0]]  # [chunk, 2, 3]
            faces_pattern1 = quads_chunk[:, QUAD2TRI[1]]  # [chunk, 2, 3]
            
            faces_chunk = torch.where(
                condition[:, None, None],
                faces_pattern0,
                faces_pattern1
            )  # [chunk, 2, 3]
            
            all_tri_faces.append(faces_chunk.reshape(-1, 3))
        
        if not all_tri_faces:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        
        return torch.cat(all_tri_faces, dim=0)
    
    def _triangulate_by_angle(
        self, 
        vertices: torch.Tensor, 
        quads: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        use_abs_normal: bool = False,
        chunk_size: int = 4096
    ) -> torch.Tensor:
        """
        Triangulate quads based on normal consistency or angle minimization.
        
        If vertex normals are provided, chooses the diagonal split that generates 
        triangles with normals most consistent with the vertex normals. 
        Otherwise falls back to angle-based splitting (minimizing deviation from 90°).
        
        Args:
            vertices: [V, 3] vertex positions
            quads: [Q, 4] quad vertex indices
            normals: [V, 3] optional vertex normals
            use_abs_normal: If True, use abs(dot) for consistency computation,
                           making it direction-agnostic (allows flipped normals)
            chunk_size: Number of quads to process per chunk (for memory efficiency)
        
        Returns:
            faces: [2*Q, 3] triangle faces
        """
        device = quads.device
        num_quads = quads.shape[0]
        
        # Two triangulation patterns
        # Pattern 0: [0,1,2], [0,2,3] - splits along 0-2 diagonal
        # Pattern 1: [0,1,3], [1,2,3] - splits along 1-3 diagonal
        QUAD2TRI = torch.tensor([
            [[0, 1, 2], [0, 2, 3]],  # Pattern 0
            [[0, 1, 3], [1, 2, 3]],  # Pattern 1
        ], device=device, dtype=torch.long)
        
        all_tri_faces = []
        
        for i in range(0, num_quads, chunk_size):
            quads_chunk = quads[i:i + chunk_size]
            quad_coords = vertices[quads_chunk]  # [chunk, 4, 3]
            
            if normals is None:
                # Angle-based: choose split that minimizes max angle deviation from 90°
                condition = self._compute_angle_condition(quad_coords)
            else:
                # Normal-based: choose split with better normal consistency
                quad_normals = normals[quads_chunk]  # [chunk, 4, 3]
                condition = self._compute_normal_consistency_condition(
                    quad_coords, quad_normals, QUAD2TRI, use_abs=use_abs_normal
                )
            
            # Apply chosen pattern
            faces_pattern0 = quads_chunk[:, QUAD2TRI[0]]  # [chunk, 2, 3]
            faces_pattern1 = quads_chunk[:, QUAD2TRI[1]]  # [chunk, 2, 3]
            
            # condition: True -> pattern0 (0-2 split), False -> pattern1 (1-3 split)
            faces_chunk = torch.where(
                condition[:, None, None],
                faces_pattern0,
                faces_pattern1
            )  # [chunk, 2, 3]
            
            # Flatten to [chunk*2, 3]
            all_tri_faces.append(faces_chunk.reshape(-1, 3))
        
        if not all_tri_faces:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        
        return torch.cat(all_tri_faces, dim=0)
    
    def _compute_angle_condition(self, quad_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute which diagonal to use based on interior angles.
        
        Choose split where the sum of opposite angles is closer to 180° (π).
        This produces better-shaped triangles.
        
        Args:
            quad_coords: [Q, 4, 3] quad vertex coordinates
        
        Returns:
            condition: [Q] bool, True if should use 0-2 diagonal
        """
        # Get edge vectors
        v0, v1, v2, v3 = quad_coords[:, 0], quad_coords[:, 1], quad_coords[:, 2], quad_coords[:, 3]
        e01 = torch.nn.functional.normalize(v1 - v0, dim=-1)
        e12 = torch.nn.functional.normalize(v2 - v1, dim=-1)
        e23 = torch.nn.functional.normalize(v3 - v2, dim=-1)
        e30 = torch.nn.functional.normalize(v0 - v3, dim=-1)
        
        # Compute interior angles using atan2 for stability
        def compute_angle(e1, e2):
            cross_norm = torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1)
            dot = (e1 * e2).sum(dim=-1)
            return torch.atan2(cross_norm, dot)
        
        angle0 = compute_angle(e01, e30)
        angle1 = compute_angle(e12, e01)
        angle2 = compute_angle(e23, e12)
        angle3 = compute_angle(e30, e23)
        
        # Sum of opposite angles
        angle_02 = angle0 + angle2  # Vertices 0 and 2
        angle_13 = angle1 + angle3  # Vertices 1 and 3
        
        # Choose diagonal that connects vertices with smaller angle sum
        # (keeps the sharper corners together in the same triangle)
        return angle_02 < angle_13
    
    def _compute_normal_consistency_condition(
        self,
        quad_coords: torch.Tensor,
        quad_normals: torch.Tensor,
        quad2tri: torch.Tensor,
        use_abs: bool = False
    ) -> torch.Tensor:
        """
        Compute which diagonal to use based on normal consistency.
        
        Choose split where generated triangle normals best match vertex normals.
        
        Args:
            quad_coords: [Q, 4, 3] quad vertex coordinates
            quad_normals: [Q, 4, 3] quad vertex normals
            quad2tri: [2, 2, 3] triangulation patterns
            use_abs: If True, use abs(dot) for consistency, allowing flipped normals
        
        Returns:
            condition: [Q] bool, True if should use 0-2 diagonal
        """
        # Generate triangles for both patterns
        tri_coords_02 = quad_coords[:, quad2tri[0]]  # [Q, 2, 3, 3]
        tri_coords_13 = quad_coords[:, quad2tri[1]]  # [Q, 2, 3, 3]
        tri_normals_02 = quad_normals[:, quad2tri[0]]  # [Q, 2, 3, 3]
        tri_normals_13 = quad_normals[:, quad2tri[1]]  # [Q, 2, 3, 3]
        
        def compute_triangle_geom_normal(coords):
            """Compute geometric normal from triangle vertices."""
            v0, v1, v2 = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = torch.cross(edge1, edge2, dim=-1)
            return normal / (normal.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        
        def compute_consistency(geom_normals, vert_normals, use_abs_dot):
            """Compute consistency as average dot product with vertex normals."""
            # geom_normals: [Q, 2, 3], vert_normals: [Q, 2, 3, 3]
            dot = (geom_normals.unsqueeze(-2) * vert_normals).sum(dim=-1)  # [Q, 2, 3]
            if use_abs_dot:
                dot = dot.abs()
            return dot.mean(dim=(-1, -2))  # [Q]
        
        # Compute geometric normals for each triangle
        geom_02 = compute_triangle_geom_normal(tri_coords_02)  # [Q, 2, 3]
        geom_13 = compute_triangle_geom_normal(tri_coords_13)  # [Q, 2, 3]
        
        # Compute consistency scores
        consistency_02 = compute_consistency(geom_02, tri_normals_02, use_abs)  # [Q]
        consistency_13 = compute_consistency(geom_13, tri_normals_13, use_abs)  # [Q]
        
        # Higher consistency = better match
        return consistency_02 > consistency_13


def decode_fct_dict(
    fct_dict: Dict[str, torch.Tensor],
    device: str = 'cuda'
) -> DecodedMesh:
    """
    Convenience function to decode from dictionary format.
    
    Args:
        fct_dict: Dictionary with keys:
            - active_voxels_indices
            - primal_anchor
            - primal_normal (optional)
            - primal_edge_flux_sign
        device: Device for computation
    
    Returns:
        DecodedMesh
    """
    # Infer resolution from indices
    max_idx = fct_dict['active_voxels_indices'].max().item()
    resolution = int((max_idx + 1) ** (1/3)) + 1
    resolution = 2 ** int(torch.log2(torch.tensor(resolution)).ceil().item())
    
    decoder = FCTDecoder(resolution=resolution, device=device)
    
    return decoder.decode(
        active_voxel_indices=fct_dict['active_voxels_indices'].to(device),
        anchors=fct_dict['primal_anchor'].to(device),
        edge_flux_sign=fct_dict.get('primal_edge_flux_sign', 
                                     torch.zeros(len(fct_dict['primal_anchor']), 12)).to(device),
        normals=fct_dict.get('primal_normal', None)
    )
