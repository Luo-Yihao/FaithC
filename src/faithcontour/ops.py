import torch
import numpy as np
import scipy.sparse as sp
from . import _C 
from .utils.grid import *

gen_candidates_overlap      = _C.gen_candidates_overlap
aabb_tri_sat_clip_select    = _C.aabb_tri_sat_clip_select
voxelize_mark               = _C.voxelize_mark
segment_tri_intersection_fused = _C.segment_tri_intersection_fused

## Not used currently:
# preprocess_tris             = _C.preprocess_tris
# seg_tri_pairs               = _C.seg_tri_pairs
# seg_tri_allpairs            = _C.seg_tri_allpairs


@torch.no_grad()
def segment_triangle_intersections_fast(
    all_seg_verts: torch.Tensor,
    all_tris_verts: torch.Tensor,
    chunk_size: int = 1024,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch implementation of segment-triangle intersection test in chunks.
    Based on the algorithm you provided, avoiding divisions for robustness.
    Processes segments in chunks to manage memory usage.
    Args:
      all_seg_verts: Float[S,2,3] - all segments' endpoints
      all_tris_verts: Float[T,3,3] - all triangles' vertices
      chunk_size: int - number of segments to process per chunk
      eps: float - small epsilon to avoid numerical issues
    Returns:
      intersecting_seg_indices: Long[K] - indices of segments that intersect any triangle
      intersecting_tri_indices: Long[K] - indices of triangles intersected by the segments
    """
    assert all_seg_verts.is_cuda and all_tris_verts.is_cuda, "Input tensors must be on a CUDA device."
    # 确保输入是 float32

    device = all_seg_verts.device
    num_segs = all_seg_verts.shape[0]
    num_tris = all_tris_verts.shape[0]

    if num_segs == 0 or num_tris == 0:
        return (torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=all_seg_verts.dtype, device=device))

    tri_aabb_min = all_tris_verts.amin(dim=1)
    tri_aabb_max = all_tris_verts.amax(dim=1)

    seg_idx, tri_idx, dots = segment_tri_intersection_fused(
        all_seg_verts.contiguous(),
        all_tris_verts.contiguous(),
        tri_aabb_min.contiguous(),
        tri_aabb_max.contiguous(),
        eps
    )

    return seg_idx, tri_idx, dots


@torch.no_grad()
def find_intersections_select(
    aabbs_min: torch.Tensor, aabbs_max: torch.Tensor, triangles: torch.Tensor,
    *,
    eps: float = 1e-8,
    max_vert: int = 7,
    aabb_chunk: int = 200_000,
    tri_chunk: int = 8_192,
    pairs_cap: int = 5_000_000,
    return_mode: str = "aabb",
    use_sat_in_fast: bool = True,
    keep_on: str = "cuda"
):
    """
    AABB Intersection with Triangles, with precise narrow-phase intersection.
    Args:
      aabbs_min: Float[N,3] - AABB minimum corners
      aabbs_max: Float[N,3] - AABB maximum corners
      triangles: Float[M,3,3] - Triangle vertices
    Keyword args:
      eps: float - small epsilon to avoid numerical issues
      max_vert: int - maximum vertices to output per polygon (for "poly" mode)
      aabb_chunk: int - number of AABBs to process per chunk
      tri_chunk: int - number of triangles to process per chunk
      pairs_cap: int - maximum candidate pairs per tile before subdivision
      return_mode: str - "aabb", "centroid", or "poly"
        "aabb": only return indices of AABBs that intersect any triangle
        "centroid": return AABB index, triangle index, intersection centroid, and area
        "poly": return AABB index, triangle index, polygon vertex count, polygon vertices, centroid, and area
      use_sat_in_fast: bool - whether to use SAT test in the fast path
      keep_on: str - "cpu" or "cuda", where to keep the results
    Returns:
      If return_mode == "aabb":
        hit_aabb_indices: Long[K] - indices of AABBs that intersect any triangle
      If return_mode == "centroid":
        hit_aabb_indices: Long[K] - indices of AABBs that intersect triangles
        hit_tri_indices:  Long[K] - indices of triangles intersected
        centroids:       Float[K,3] - centroids of intersection polygons
        areas:           Float[K]   - areas of intersection polygons
      If return_mode == "poly":
        hit_aabb_indices: Long[K] - indices of AABBs that intersect triangles
        hit_tri_indices:  Long[K] - indices of triangles intersected
        poly_counts:     Int32[K] - number of vertices per intersection polygon
        poly_verts:      Float[K,max_vert,3] - vertices of intersection polygons (padded)
        centroids:       Float[K,3] - centroids of intersection polygons
        areas:           Float[K]   - areas of intersection polygons
    Note:
      - triangles must be on CUDA; aabbs_min and aabbs_max can be on CPU or CUDA
      - if keep_on="cpu", results are moved to CPU before returning
      - empty inputs are handled gracefully
    """
    assert triangles.is_cuda, "triangles must be on CUDA"
    device = triangles.device
    dtype  = triangles.dtype
    N = aabbs_min.shape[0]
    M = triangles.shape[0]

    aabbs_min_d = aabbs_min.to(device, non_blocking=True)
    aabbs_max_d = aabbs_max.to(device, non_blocking=True)

    def _maybe_move(*tensors):
        if keep_on == "cpu":
            return tuple(x.detach().cpu() for x in tensors)
        return tensors

    # Fast path for empty inputs
    if N == 0 or M == 0:
        if return_mode == "aabb":
            out = torch.empty(0, dtype=torch.long, device=device)
            return _maybe_move(out)[0]
        elif return_mode == "centroid":
            zlong = torch.empty(0, dtype=torch.long, device=device)
            out = (zlong, zlong,
                   torch.empty(0,3, dtype=dtype, device=device),
                   torch.empty(0,   dtype=dtype, device=device))
            return _maybe_move(*out)
        else:  # "poly"
            zlong = torch.empty(0, dtype=torch.long, device=device)
            out = (zlong, zlong,
                   torch.empty(0, dtype=torch.int32, device=device),
                   torch.empty(0, max_vert, 3, dtype=dtype, device=device),
                   torch.empty(0, 3, dtype=dtype, device=device),
                   torch.empty(0,   dtype=dtype, device=device))
            return _maybe_move(*out)

    # Establish triangle AABBs
    tri_min_full = triangles.amin(dim=1)
    tri_max_full = triangles.amax(dim=1)

    # Fast path for "aabb" mode only
    if return_mode == "aabb":
        active_mask = torch.zeros((N,), dtype=torch.uint8, device=device)
        for a0 in range(0, N, aabb_chunk):
            a1 = min(a0 + aabb_chunk, N)
            a_min = aabbs_min_d[a0:a1].contiguous()
            a_max = aabbs_max_d[a0:a1].contiguous()
            for t0 in range(0, M, tri_chunk):
                t1 = min(t0 + tri_chunk, M)
                t_min = tri_min_full[t0:t1].contiguous()
                t_max = tri_max_full[t0:t1].contiguous()
                tris  = triangles[t0:t1].contiguous()
                voxelize_mark(
                    a_min, a_max, t_min, t_max, tris,
                    a0, t0, active_mask, bool(use_sat_in_fast), float(eps)
                )
        hit_ids = torch.nonzero(active_mask, as_tuple=False).view(-1).long()
        if hit_ids.numel():
            hit_ids, _ = torch.sort(hit_ids)
        return _maybe_move(hit_ids)[0]

    # Narrow phase setup
    if return_mode == "centroid":
        A_all, T_all, C_all, S_all = [], [], [], []
        mode_id = 1
    else:  # "poly"
        A_all, T_all, PC_all, PV_all, C_all, S_all = [], [], [], [], [], []
        mode_id = 2

    # Tile processing with recursive subdivision
    def process_tile(a_min, a_max, a_off, t_min, t_max, t_off):
        # Local candidate buffer
        cap = pairs_cap
        cand_a = torch.empty((cap,), device=device, dtype=torch.long)
        cand_t = torch.empty((cap,), device=device, dtype=torch.long)
        counter  = torch.zeros((1,), device=device, dtype=torch.int32)
        overflow = torch.zeros((1,), device=device, dtype=torch.uint8)

        # Try once
        counter.zero_(); overflow.zero_()
        gen_candidates_overlap(
            a_min, a_max, t_min, t_max,
            int(a_off), int(t_off),
            cand_a, cand_t,
            counter, overflow, float(eps)
        )

        if bool(overflow.item()):
            # 溢出：沿长边二分
            Na = a_min.shape[0]
            Nt = t_min.shape[0]
            if Na >= Nt and Na > 1:
                mid = Na // 2
                process_tile(a_min[:mid], a_max[:mid], a_off,
                             t_min, t_max, t_off)
                process_tile(a_min[mid:], a_max[mid:], a_off+mid,
                             t_min, t_max, t_off)
            elif Nt > 1:
                mid = Nt // 2
                process_tile(a_min, a_max, a_off,
                             t_min[:mid], t_max[:mid], t_off)
                process_tile(a_min, a_max, a_off,
                             t_min[mid:], t_max[mid:], t_off+mid)
            else:
                # Fallback to non-overlapping case
                K = int(counter.item()); 
                if K > cap: K = cap
                if K > 0:
                    idx_global = cand_a[:K]          # Global a index
                    a_loc_min  = aabbs_min_d.index_select(0, idx_global).contiguous()
                    a_loc_max  = aabbs_max_d.index_select(0, idx_global).contiguous()
                    cand_a_loc = torch.arange(K, device=device, dtype=torch.long)  # Local 0..K-1
                    cand_t_idx = cand_t[:K]
                    hit_mask, out_a_loc, out_t_idx, poly_counts, poly_verts, cents, areas = aabb_tri_sat_clip_select(
                        a_loc_min, a_loc_max, triangles, cand_a_loc, cand_t_idx, mode_id, float(eps), int(max_vert)
                    )
                    # CUDA kernel may return empty hit_mask
                    if hit_mask.numel():
                        mask = hit_mask
                    else:
                        mask = torch.ones_like(out_a_loc, dtype=torch.bool, device=device)

                    if mode_id == 1:
                        A_all.append(idx_global[out_a_loc][mask])
                        T_all.append(out_t_idx[mask])
                        C_all.append(cents[mask])
                        S_all.append(areas[mask])
                    else:
                        A_all.append(idx_global[out_a_loc][mask])
                        T_all.append(out_t_idx[mask])
                        PC_all.append(poly_counts[mask])
                        PV_all.append(poly_verts[mask])
                        C_all.append(cents[mask])
                        S_all.append(areas[mask])
            return

        # No overflow: process candidates
        K = int(counter.item())
        if K <= 0:
            return
        idx_global = cand_a[:K]          # Global a index
        a_loc_min  = aabbs_min_d.index_select(0, idx_global).contiguous()
        a_loc_max  = aabbs_max_d.index_select(0, idx_global).contiguous()
        cand_a_loc = torch.arange(K, device=device, dtype=torch.long)  # Local 0..K-1
        cand_t_idx = cand_t[:K]
        hit_mask, out_a_loc, out_t_idx, poly_counts, poly_verts, cents, areas = aabb_tri_sat_clip_select(
            a_loc_min, a_loc_max, triangles, cand_a_loc, cand_t_idx, mode_id, float(eps), int(max_vert)
        )
        if hit_mask.numel():
            mask = hit_mask
        else:
            mask = torch.ones_like(out_a_loc, dtype=torch.bool, device=device)

        if mode_id == 1:
            A_all.append(idx_global[out_a_loc][mask])
            T_all.append(out_t_idx[mask])
            C_all.append(cents[mask])
            S_all.append(areas[mask])
        else:
            A_all.append(idx_global[out_a_loc][mask])
            T_all.append(out_t_idx[mask])
            PC_all.append(poly_counts[mask])
            PV_all.append(poly_verts[mask])
            C_all.append(cents[mask])
            S_all.append(areas[mask])

    # Top-level tiling
    for a0 in range(0, N, aabb_chunk):
        a1 = min(a0 + aabb_chunk, N)
        a_min = aabbs_min_d[a0:a1].contiguous()
        a_max = aabbs_max_d[a0:a1].contiguous()
        for t0 in range(0, M, tri_chunk):
            t1 = min(t0 + tri_chunk, M)
            t_min = tri_min_full[t0:t1].contiguous()
            t_max = tri_max_full[t0:t1].contiguous()
            process_tile(a_min, a_max, a0, t_min, t_max, t0)

    # Gather results
    if return_mode == "centroid":
        if not A_all:
            zlong = torch.empty(0, dtype=torch.long, device=device)
            out = (zlong, zlong,
                   torch.empty(0,3, dtype=dtype, device=device),
                   torch.empty(0,   dtype=dtype, device=device))
            return _maybe_move(*out)
        aidx = torch.cat(A_all); tidx = torch.cat(T_all)
        cents= torch.cat(C_all); areas= torch.cat(S_all)
        return _maybe_move(aidx, tidx, cents, areas)
    else:  # "poly"
        if not A_all:
            zlong = torch.empty(0, dtype=torch.long, device=device)
            out = (zlong, zlong,
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, max_vert, 3, dtype=dtype, device=device),
                torch.empty(0, 3, dtype=dtype, device=device),
                torch.empty(0,   dtype=dtype, device=device))
            return _maybe_move(*out)
        aidx = torch.cat(A_all); tidx = torch.cat(T_all)
        pc   = torch.cat(PC_all); pv = torch.cat(PV_all)
        cents= torch.cat(C_all);  areas = torch.cat(S_all)
        return _maybe_move(aidx, tidx, pc, pv, cents, areas)
        


@torch.no_grad()
def solve_points_by_group(
    group_idx: torch.Tensor,       # [K], group IDs, e.g., aabb_idx
    points: torch.Tensor,          # [K, 3], data points, e.g., centroids
    normals_per_point: torch.Tensor, # [K, 3], unit normals for each point
    weights_per_point: torch.Tensor, # [K], weights for each point (e.g., areas)
    *,
    lambda_n: float = 1.0,
    lambda_d: float = 1.0,
    weight_power: float = 1.0,     # exponent for weights
    eps: float = 1e-12
):
    """
    Group a set of points and solve for an optimal point per group using the same logic as average_point_from_points_and_normals.

    This is a pure, direct batched implementation of avg_point, designed for scenarios where:
    - The input points, normals, weights, and group IDs are all [K] tensors and correspond one-to-one.
    - No indirect lookup via tri_idx.
    - Uses the same math as avg_point (group-wise weight normalization, fixed regularization, linalg.solve).

    Returns:
      unique_group_ids: Long[V]    — all unique group IDs (sorted, unique)
      p_solutions:      Float[V,3] — optimal point for each group
      n_solutions:      Float[V,3] — weighted average normal for each group
    """
    # ---- Device and dtype ----
    target_dev = points.device
    dtype = points.dtype

    # ---- Validation ----
    K = group_idx.numel()
    if not (points.shape[0] == K and normals_per_point.shape[0] == K and weights_per_point.numel() == K):
        raise ValueError("Inputs group_idx, points, normals_per_point, and weights_per_point must have the same length K.")
    if K == 0:
        return (torch.empty(0, dtype=torch.long, device=target_dev),
                torch.empty(0, 3, dtype=dtype, device=target_dev),
                torch.empty(0, 3, dtype=dtype, device=target_dev))

    # ---- Step 1: Grouping ----
    unique_group_ids, inv, counts = torch.unique(group_idx.cuda(),
                                                 sorted=True, 
                                                 return_inverse=True, 
                                                 return_counts=True)
    V = unique_group_ids.numel()

    # ---- Step 2: Compute and normalize weights per group ----
    W_raw = weights_per_point.clamp_min(0).pow(weight_power)
    sum_per_group = torch.zeros(V, dtype=dtype, device=target_dev)
    sum_per_group.index_add_(0, inv, W_raw)
    
    sum_per_group_inv = sum_per_group[inv]
    W = torch.where(
        sum_per_group_inv > eps,
        W_raw / sum_per_group_inv,
        (1.0 / counts[inv]).to(dtype) # Use uniform weights if group sum is zero
    )

    # ---- Step 3: Construct A and b matrices ----
    I3 = torch.eye(3, device=target_dev, dtype=dtype)
    # n*n^T, shape [K, 3, 3]
    outer = normals_per_point.unsqueeze(2) * normals_per_point.unsqueeze(1)
    # W_i = (lambda_d * I + lambda_n * n_i*n_i^T), shape [K, 3, 3]
    W_i = lambda_d * I3 + lambda_n * outer

    # A_i = w_i * W_i, shape [K, 3, 3]
    A_i = W.view(K, 1, 1) * W_i
    # b_i = w_i * W_i @ x_i, shape [K, 3]
    b_i = (A_i @ points.unsqueeze(-1)).squeeze(-1)

    # ---- Step 4: Aggregate A and b per group ----
    A_group = torch.zeros(V, 3, 3, device=target_dev, dtype=dtype)
    b_group = torch.zeros(V, 3,    device=target_dev, dtype=dtype)
    A_group.index_add_(0, inv, A_i)
    b_group.index_add_(0, inv, b_i)

    # ---- Step 5: Solve Ap=b for each group ----
    # Use the same fixed regularization as avg_point
    A_reg = A_group + (eps * I3)
    # Directly solve, will raise LinAlgError if singular
    p_solutions = torch.linalg.solve(A_reg, b_group)

    # ---- (Optional) Compute weighted average normal per group ----
    nsum = torch.zeros(V, 3, device=target_dev, dtype=dtype)
    nsum.index_add_(0, inv, W.view(K, 1) * normals_per_point)
    n_solutions = nsum / nsum.norm(dim=-1, keepdim=True).clamp_min(eps)

    return unique_group_ids, p_solutions, n_solutions
