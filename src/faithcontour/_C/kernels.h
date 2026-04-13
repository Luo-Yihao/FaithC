#pragma once
#include <torch/extension.h>
#include <vector>

void gen_candidates_overlap_cuda(
    at::Tensor aabb_min, at::Tensor aabb_max,
    at::Tensor tri_min,  at::Tensor tri_max,
    int64_t a_offset, int64_t t_offset,
    at::Tensor cand_a_out, at::Tensor cand_t_out,
    at::Tensor counter, at::Tensor overflow, double eps);

std::vector<at::Tensor> aabb_tri_sat_clip_select_cuda(
    at::Tensor aabbs_min, at::Tensor aabbs_max,
    at::Tensor tris_verts,
    at::Tensor cand_a_idx, at::Tensor cand_t_idx,
    int64_t mode, double eps, int64_t max_vert);

void voxelize_mark_cuda(
    at::Tensor aabb_min, at::Tensor aabb_max,
    at::Tensor tri_min,  at::Tensor tri_max,
    at::Tensor tris_verts,
    int64_t a_offset, int64_t t_offset,
    at::Tensor active_mask,
    bool use_sat, double eps);

std::vector<at::Tensor> segment_tri_intersection_fused_cuda(
    at::Tensor seg_verts,
    at::Tensor tris_verts,
    at::Tensor tri_aabb_min,
    at::Tensor tri_aabb_max,
    double eps);