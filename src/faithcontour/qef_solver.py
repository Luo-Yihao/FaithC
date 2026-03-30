"""
QEF (Quadric Error Function) Solver for anchor point computation.

Provides both standard and differentiable versions.
"""

import torch
from torch_scatter import scatter_sum, scatter_mean
from typing import Tuple, Optional


def solve_qef(
    group_ids: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    weights: torch.Tensor,
    *,
    lambda_n: float = 1.0,
    lambda_d: float = 0.1,
    weight_power: float = 1.0,
    cell_size: float = 1.0,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve for optimal anchor points using QEF (Quadric Error Function).

    Given a set of sample points with normals and weights, grouped by voxel,
    solves for the optimal representative point per voxel.

    Internally works in voxel-relative coordinates (centered at group centroid,
    scaled by 1/cell_size) for better float32 precision at high resolutions.

    Args:
        group_ids: [K] voxel assignment for each sample
        points: [K, 3] sample positions (e.g., SAT clip centroids)
        normals: [K, 3] sample normals (unit vectors)
        weights: [K] sample weights (e.g., clip areas)
        lambda_n: Normal constraint weight
        lambda_d: Distance regularization weight
        weight_power: Exponent for weights
        cell_size: Voxel size for coordinate normalization
        eps: Numerical stability epsilon

    Returns:
        unique_ids: [V] unique voxel IDs
        anchors: [V, 3] anchor point per voxel
        anchor_normals: [V, 3] average normal per voxel

    Note:
        This is a non-differentiable version using torch.linalg.solve.
        For differentiable version, use solve_qef_differentiable.
    """
    device = points.device
    dtype = points.dtype
    K = group_ids.numel()
    
    if K == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 3, dtype=dtype, device=device),
            torch.empty(0, 3, dtype=dtype, device=device)
        )
    
    # Get unique groups
    unique_ids, inverse, counts = torch.unique(
        group_ids, sorted=True, return_inverse=True, return_counts=True
    )
    V = unique_ids.numel()
    
    # Normalize weights per group
    raw_weights = weights.clamp_min(0).pow(weight_power)
    weight_sum_per_group = scatter_sum(raw_weights, inverse, dim_size=V)

    normalized_weights = torch.where(
        weight_sum_per_group[inverse] > eps,
        raw_weights / weight_sum_per_group[inverse],
        (1.0 / counts[inverse]).to(dtype)
    )

    # Transform to voxel-relative coordinates for numerical precision.
    # At res 2048 in float32, global coords have only ~3 significant digits
    # within a voxel. Centering + scaling to unit voxel preserves full precision.
    centroids = scatter_mean(points, inverse, dim=0, dim_size=V)  # [V, 3]
    inv_cell = 1.0 / cell_size
    points_local = (points - centroids[inverse]) * inv_cell  # [K, 3] in ~[-0.5, 0.5]

    # Build QEF matrices: A @ p_local = b
    # W_i = lambda_d * I + lambda_n * (n_i @ n_i^T)
    I3 = torch.eye(3, device=device, dtype=dtype)
    outer = normals.unsqueeze(2) * normals.unsqueeze(1)  # [K, 3, 3]
    W_i = lambda_d * I3 + lambda_n * outer

    # Weighted contributions
    A_i = normalized_weights.view(K, 1, 1) * W_i  # [K, 3, 3]
    b_i = (A_i @ points_local.unsqueeze(-1)).squeeze(-1)  # [K, 3]

    # Aggregate per group
    A_group = torch.zeros(V, 3, 3, device=device, dtype=dtype)
    b_group = torch.zeros(V, 3, device=device, dtype=dtype)

    # Scatter add
    A_group.scatter_add_(0, inverse.view(-1, 1, 1).expand(-1, 3, 3), A_i)
    b_group.scatter_add_(0, inverse.view(-1, 1).expand(-1, 3), b_i)

    # Solve with regularization
    A_reg = A_group + eps * I3
    anchors_local = torch.linalg.solve(A_reg, b_group)  # [V, 3] in voxel-relative
    anchors = anchors_local * cell_size + centroids  # back to global
    
    # Compute average normals
    weighted_normals = scatter_sum(
        normalized_weights.view(K, 1) * normals, 
        inverse, 
        dim=0, 
        dim_size=V
    )
    anchor_normals = weighted_normals / (weighted_normals.norm(dim=1, keepdim=True).clamp_min(eps))
    
    return unique_ids, anchors, anchor_normals


def solve_qef_differentiable(
    group_ids: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable anchor solver using weighted average (FlexiCubes-style).
    
    This version supports gradient backpropagation through the anchor computation.
    
    Args:
        group_ids: [K] voxel assignment for each sample
        points: [K, 3] sample positions
        weights: [K] sample weights (can be learnable)
        eps: Numerical stability epsilon
    
    Returns:
        unique_ids: [V] unique voxel IDs
        anchors: [V, 3] anchor point per voxel
    
    Note:
        Gradients flow through both `points` and `weights`.
    """
    device = points.device
    dtype = points.dtype
    K = group_ids.numel()
    
    if K == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 3, dtype=dtype, device=device)
        )
    
    # Get unique groups
    unique_ids, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    V = unique_ids.numel()
    
    # Weighted sum
    weights_pos = weights.clamp_min(eps)
    weighted_points = weights_pos.view(K, 1) * points
    
    sum_weighted_points = scatter_sum(weighted_points, inverse, dim=0, dim_size=V)
    sum_weights = scatter_sum(weights_pos, inverse, dim_size=V)
    
    anchors = sum_weighted_points / sum_weights.view(V, 1).clamp_min(eps)
    
    return unique_ids, anchors
