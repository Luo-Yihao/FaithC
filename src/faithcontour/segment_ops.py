"""
Segment-triangle intersection utilities with dot product computation.

Wraps Atom3d's intersect_segment for FCT edge flux calculation.
"""

import torch
from typing import Tuple, Optional
from atom3d import MeshBVH


def intersect_segments_with_dot(
    bvh: MeshBVH,
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Segment-mesh intersection with dot product of edge direction and face normal.
    
    Used for computing edge flux signs in FCT encoding.
    
    Args:
        bvh: MeshBVH instance
        seg_start: [N, 3] segment start points
        seg_end: [N, 3] segment end points
    
    Returns:
        valid_mask: [K] indices of segments that intersect (into original N)
        face_ids: [K] face IDs of intersection
        dots: [K] dot product of segment direction with face normal
              Positive = segment points in same direction as face normal
              Negative = segment points opposite to face normal
    """
    device = seg_start.device
    N = seg_start.shape[0]
    
    # Call Atom3d's segment intersection
    result = bvh.intersect_segment(seg_start, seg_end)
    
    # Get valid hits
    valid_indices = torch.where(result.hit)[0]
    
    if valid_indices.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=seg_start.dtype, device=device)
        )
    
    # Extract valid data
    valid_face_ids = result.face_ids[valid_indices]
    valid_start = seg_start[valid_indices]
    valid_end = seg_end[valid_indices]
    
    # Compute segment directions (normalized)
    seg_dir = valid_end - valid_start
    seg_dir = seg_dir / (seg_dir.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute face normals for hit faces
    face_normals = _compute_face_normals(bvh, valid_face_ids)
    
    # Compute dot products
    dots = (seg_dir * face_normals).sum(dim=1)
    
    return valid_indices, valid_face_ids, dots


def _compute_face_normals(bvh: MeshBVH, face_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute face normals for given face IDs.
    
    Args:
        bvh: MeshBVH instance
        face_ids: [K] face indices
    
    Returns:
        normals: [K, 3] unit face normals
    """
    # Get triangle vertices
    tri_verts = bvh.vertices[bvh.faces[face_ids.long()]]  # [K, 3, 3]
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    
    # Cross product for normal
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    
    return normals


def compute_edge_flux_sign(
    bvh: MeshBVH,
    edge_endpoints: torch.Tensor,
) -> torch.Tensor:
    """
    Compute edge flux signs for FCT encoding.
    
    For each edge, determines if the surface crosses it and in which direction.
    
    Args:
        bvh: MeshBVH instance
        edge_endpoints: [E, 2, 3] edge endpoint coordinates
    
    Returns:
        flux_sign: [E] in {-1, 0, +1}
            0 = no crossing
            +1 = surface normal aligns with edge direction
            -1 = surface normal opposes edge direction
    """
    E = edge_endpoints.shape[0]
    device = edge_endpoints.device
    
    seg_start = edge_endpoints[:, 0]
    seg_end = edge_endpoints[:, 1]
    
    valid_indices, face_ids, dots = intersect_segments_with_dot(bvh, seg_start, seg_end)
    
    # Initialize all edges as no crossing
    flux_sign = torch.zeros(E, dtype=torch.int8, device=device)
    
    if valid_indices.numel() > 0:
        # For edges with multiple crossings, use the one with minimal |dot|
        # (most perpendicular to the surface)
        try:
            from torch_scatter import scatter_min
        except ImportError:
            from .torch_scatter_compat import scatter_min
        
        _, best_idx = scatter_min(dots.abs(), valid_indices, dim_size=E)
        
        # Get the sign of the best dot
        all_dots = torch.cat([dots, torch.zeros(1, device=device)])
        best_dots = all_dots[best_idx]
        
        flux_sign = best_dots.sign().to(torch.int8)
    
    return flux_sign
