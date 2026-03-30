"""
FCT Encoder: Mesh → Faithful Contour Token

Encodes a mesh into a compact representation for each active voxel:
- Anchor point (surface sample)
- Normal direction
- Edge flux signs (12 edges per voxel)
"""

import torch
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass, field

from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer

from .qef_solver import solve_qef
from .segment_ops import compute_edge_flux_sign


@dataclass
class FCTResult:
    """Result of FCT encoding."""
    
    # Active voxel indices at max resolution
    active_voxel_indices: torch.Tensor  # [K] linear indices
    
    # Per-voxel anchor point (surface representative)
    anchor: torch.Tensor  # [K, 3]
    
    # Per-voxel surface normal
    normal: torch.Tensor  # [K, 3]
    
    # Per-voxel edge flux signs: {-1, 0, +1}
    edge_flux_sign: torch.Tensor  # [K, 12]
    
    # Resolution info
    resolution: int = 0
    max_level: int = 0


class FCTEncoder:
    """
    Faithful Contour Token Encoder.
    
    Encodes a mesh into per-voxel surface tokens using Atom3d primitives.
    
    Pipeline:
    1. Hierarchical octree traversal → active voxels
    2. SAT polygon clipping → centroids, areas, normals per (voxel, triangle) pair
    3. QEF solve → anchor points and normals per voxel
    4. Edge flux computation → surface crossing directions
    
    Example:
        bvh = MeshBVH(vertices, faces)
        octree = OctreeIndexer(max_level=7, bounds=bvh.get_bounds())
        
        encoder = FCTEncoder(bvh, octree)
        result = encoder.encode()
        
        # result.anchor: [K, 3]
        # result.normal: [K, 3]
        # result.edge_flux_sign: [K, 12]
    """
    
    def __init__(
        self,
        bvh: MeshBVH,
        octree: OctreeIndexer,
        device: str = 'cuda'
    ):
        """
        Args:
            bvh: MeshBVH instance for the mesh
            octree: OctreeIndexer with bounds matching mesh
            device: Device for computation
        """
        self.bvh = bvh
        self.octree = octree
        self.device = device
        
        # Precompute face normals
        self._face_normals = self._compute_face_normals()
    
    def _compute_face_normals(self) -> torch.Tensor:
        """Compute unit face normals for all triangles."""
        tri_verts = self.bvh.vertices[self.bvh.faces]  # [F, 3, 3]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        
        normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        normals = normals / (normals.norm(dim=1, keepdim=True).clamp_min(1e-8))
        
        return normals
    
    def encode(
        self,
        min_level: int = 2,
        solver_weights: Optional[Dict[str, float]] = None,
        compute_flux: bool = True,
        clamp_anchors: bool = False
    ) -> FCTResult:
        """
        Encode mesh into FCT representation.
        
        Args:
            min_level: Starting octree level for traversal
            solver_weights: QEF solver parameters {lambda_n, lambda_d, weight_power}
            compute_flux: Whether to compute edge flux signs
            clamp_anchors: Whether to clamp anchors to voxel bounds and project to surface
        
        Returns:
            FCTResult with anchor, normal, and edge_flux_sign per active voxel
        """
        if solver_weights is None:
            solver_weights = {
                'lambda_n': 1.0,
                'lambda_d': 0.1,
                'weight_power': 1.0
            }
        
        # Step 1 & 2: Combined octree traversal + SAT clip
        active_voxel_ijk, clip_data = self._octree_traverse_with_clip(min_level)
        
        if active_voxel_ijk.numel() == 0 or clip_data['centroids'].numel() == 0:
            return self._empty_result()
        
        # Convert to linear indices
        active_voxel_idx = self.octree.ijk_to_cube(active_voxel_ijk)
        
        # Step 3: QEF solve for anchors and normals
        cell_size = self.octree.get_cell_size(self.octree.max_level)
        cell_size_scalar = cell_size[0].item()  # uniform grid, all axes equal
        unique_ids, anchors, normals = solve_qef(
            group_ids=clip_data['aabb_ids'],
            points=clip_data['centroids'],
            normals=clip_data['normals'],
            weights=clip_data['areas'],
            cell_size=cell_size_scalar,
            **solver_weights
        )
        
        # Map back to active_voxel_idx ordering
        # unique_ids contains indices into active_voxel_idx
        final_voxel_idx = active_voxel_idx[unique_ids]
        
        # Step 3.5: BVH surface projection and voxel constraint
        if clamp_anchors:
            anchors = self._clamp_and_project_anchors(anchors, final_voxel_idx)
        
        # Step 4: Compute edge flux signs
        if compute_flux:
            edge_flux_sign = self._compute_edge_flux(final_voxel_idx)
        else:
            edge_flux_sign = torch.zeros(
                len(final_voxel_idx), 12, 
                dtype=torch.int8, 
                device=self.device
            )
        
        return FCTResult(
            active_voxel_indices=final_voxel_idx,
            anchor=anchors,
            normal=normals,
            edge_flux_sign=edge_flux_sign,
            resolution=self.octree.res,
            max_level=self.octree.max_level
        )
    
    def _clamp_and_project_anchors(
        self,
        anchors: torch.Tensor,
        voxel_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Clamp anchors to voxel bounds and project to nearest surface.
        
        Pipeline:
        1. Clamp anchors to their voxel AABBs
        2. If clamped, project to nearest surface point using BVH UDF
        3. Re-clamp to ensure anchors stay within voxel bounds
        
        Args:
            anchors: [K, 3] anchor points
            voxel_indices: [K] linear voxel indices
        
        Returns:
            anchors_refined: [K, 3] refined anchor points
        """
        device = self.device
        K = anchors.shape[0]
        
        # Get voxel AABBs using cube_aabb_level (OctreeIndexer API)
        voxel_min, voxel_max = self.octree.cube_aabb_level(voxel_indices)
        voxel_min = voxel_min.to(device)
        voxel_max = voxel_max.to(device)
        
        # Step 1: Clamp to voxel AABB
        anchors_clamped = torch.clamp(anchors, voxel_min, voxel_max)
        
        # Check which anchors were moved
        needs_projection = (anchors_clamped - anchors).norm(dim=-1) > 1e-8
        
        # Step 2: Project clamped anchors to surface using BVH UDF
        if needs_projection.any():
            proj_idx = torch.where(needs_projection)[0]

            # Query UDF only for anchors that need projection
            udf_result = self.bvh.udf(
                anchors_clamped[proj_idx],
                return_closest=True,
                return_uvw=True,
                return_face_ids=True
            )

            # Clamp barycentric coords to stay inside triangle (avoid edge artifacts)
            uvw_clamped = udf_result.uvw.clamp(min=1e-4, max=1.0 - 1e-4)
            uvw_clamped = uvw_clamped / uvw_clamped.sum(dim=-1, keepdim=True)

            # Recompute closest point with clamped barycentrics
            triangles = self.bvh.vertices[self.bvh.faces[udf_result.face_ids.long()]]
            projected = (triangles * uvw_clamped.unsqueeze(-1)).sum(dim=1)

            # Write back only projected anchors
            anchors_refined = anchors_clamped.clone()
            anchors_refined[proj_idx] = projected
        else:
            anchors_refined = anchors_clamped
        
        # Step 3: Get voxel centers and compute relative position
        voxel_centers = (voxel_min + voxel_max) / 2.0
        half_size = (voxel_max - voxel_min) / 2.0
        
        # Clamp relative to voxel center to ensure within bounds
        anchors_rel = anchors_refined - voxel_centers
        anchors_rel = torch.clamp(anchors_rel, -half_size, half_size)
        anchors_final = anchors_rel + voxel_centers
        
        return anchors_final
    
    def _octree_traverse_with_clip(
        self, 
        min_level: int,
        broadphase_chunk_size: int = 65536,
        clip_chunk_size: int = 16384
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical octree traversal with SAT clip at final level.
        
        Uses chunked processing to avoid OOM on large meshes.
        
        Args:
            min_level: Starting octree level
            broadphase_chunk_size: Chunk size for broadphase intersection
            clip_chunk_size: Chunk size for SAT clipping (smaller due to more memory)
        
        Returns:
            active_voxel_ijk: [K, 3] active voxel coordinates
            clip_data: Dict with centroids, areas, normals, aabb_ids, face_ids
        """
        device = self.device
        empty_clip = {
            'aabb_ids': torch.empty(0, dtype=torch.long, device=device),
            'face_ids': torch.empty(0, dtype=torch.int32, device=device),
            'centroids': torch.empty(0, 3, device=device),
            'areas': torch.empty(0, device=device),
            'normals': torch.empty(0, 3, device=device),
        }
        
        # Start with all cubes at min_level
        current_ijk = self.octree.all_cubes_at_level(min_level)
        
        # Hierarchical refinement
        for level in range(min_level, self.octree.max_level + 1):
            if current_ijk.numel() == 0:
                return torch.empty(0, 3, dtype=torch.long, device=device), empty_clip
            
            if level < self.octree.max_level:
                # Intermediate levels: use broadphase (mode=0) with chunking
                chunk_size = broadphase_chunk_size
                active_mask = torch.zeros(current_ijk.shape[0], dtype=torch.bool, device=device)
                
                for i in range(0, current_ijk.shape[0], chunk_size):
                    chunk_ijk = current_ijk[i:i + chunk_size]
                    cube_min, cube_max = self.octree.cube_aabb_level(chunk_ijk, level)
                    result = self.bvh.intersect_aabb(cube_min, cube_max, mode=0)
                    active_mask[i:i + chunk_size] = result.hit
                
                active_ijk = current_ijk[active_mask]
                
                # Subdivide active cubes for next level
                if active_ijk.numel() > 0:
                    current_ijk = self.octree.subdivide(active_ijk, level)
                else:
                    current_ijk = torch.empty(0, 3, dtype=torch.long, device=device)
            else:
                # Final level: use SAT clip (mode=2) with chunking
                chunk_size = clip_chunk_size
                all_aabb_ids = []
                all_face_ids = []
                all_centroids = []
                all_areas = []
                all_normals = []
                
                for i in range(0, current_ijk.shape[0], chunk_size):
                    chunk_ijk = current_ijk[i:i + chunk_size]
                    cube_min, cube_max = self.octree.cube_aabb_level(chunk_ijk, level)
                    result = self.bvh.intersect_aabb(cube_min, cube_max, mode=2)
                    
                    if hasattr(result, 'aabb_ids') and result.aabb_ids is not None and result.aabb_ids.numel() > 0:
                        # Offset aabb_ids by chunk start index
                        all_aabb_ids.append(result.aabb_ids + i)
                        all_face_ids.append(result.face_ids)
                        all_centroids.append(result.centroids)
                        all_areas.append(result.areas)
                        all_normals.append(self._face_normals[result.face_ids.long()])
                
                if not all_aabb_ids:
                    return torch.empty(0, 3, dtype=torch.long, device=device), empty_clip
                
                # Concatenate all chunks
                aabb_ids = torch.cat(all_aabb_ids, dim=0)
                face_ids = torch.cat(all_face_ids, dim=0)
                centroids = torch.cat(all_centroids, dim=0)
                areas = torch.cat(all_areas, dim=0)
                normals = torch.cat(all_normals, dim=0)
                
                # Get unique active voxels and create remapping
                active_indices, inverse_mapping = torch.unique(aabb_ids, return_inverse=True)
                current_ijk = current_ijk[active_indices.long()]
                
                clip_data = {
                    'aabb_ids': inverse_mapping.long(),  # Now in range [0, num_active)
                    'face_ids': face_ids,
                    'centroids': centroids,
                    'areas': areas,
                    'normals': normals,
                }
                
                return current_ijk, clip_data
        
        return torch.empty(0, 3, dtype=torch.long, device=device), empty_clip
    
    def _compute_edge_flux(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute edge flux signs for active voxels.
        
        For each of the 12 edges per voxel, determines if the surface
        crosses it and the crossing direction.
        
        This method accounts for edge direction to ensure flux signs are
        consistent with local edge topology. When adjacent voxels share
        an edge but reference it in opposite directions, the flux sign
        is corrected using the edge direction flag.
        """
        K = voxel_indices.shape[0]
        
        # Get unique edges - edges are canonically defined, no direction concept
        unique_edges, voxel_to_edge = self.octree.voxel_unique_edges(voxel_indices)
        
        # Get edge endpoints for flux computation
        edge_endpoints = self.octree.edge_endpoints(unique_edges)
        
        # Compute flux for unique edges using canonical direction
        # flux_sign = sign(dot(edge_direction, surface_normal))
        unique_flux = compute_edge_flux_sign(self.bvh, edge_endpoints)  # [E]
        
        # Map back to per-voxel edges
        # Each (voxel, local_edge) pair maps to the same unique edge with same flux
        edge_flux = unique_flux[voxel_to_edge]  # [K, 12]
        
        return edge_flux.to(torch.int8)
    
    def _empty_result(self) -> FCTResult:
        """Return empty FCT result."""
        return FCTResult(
            active_voxel_indices=torch.empty(0, dtype=torch.long, device=self.device),
            anchor=torch.empty(0, 3, device=self.device),
            normal=torch.empty(0, 3, device=self.device),
            edge_flux_sign=torch.empty(0, 12, dtype=torch.int8, device=self.device),
            resolution=self.octree.res,
            max_level=self.octree.max_level
        )
    
    def to_token(self, result: FCTResult) -> torch.Tensor:
        """
        Convert FCT result to dense token format [K, D].
        
        Token format (D=19):
        - [0:3]: anchor (3)
        - [3:6]: normal (3)
        - [6:18]: edge_flux_sign (12)
        - [18]: voxel index (1)
        
        This format is suitable for neural network input.
        """
        K = result.anchor.shape[0]
        
        token = torch.cat([
            result.anchor,  # [K, 3]
            result.normal,  # [K, 3]
            result.edge_flux_sign.float(),  # [K, 12]
            result.active_voxel_indices.float().unsqueeze(-1),  # [K, 1]
        ], dim=-1)
        
        return token  # [K, 19]
