#include <torch/extension.h>
#include "kernels.h"

void gen_candidates_overlap(
    at::Tensor aabb_min, at::Tensor aabb_max,
    at::Tensor tri_min,  at::Tensor tri_max,
    int64_t a_offset, int64_t t_offset,
    at::Tensor cand_a_out, at::Tensor cand_t_out,
    at::Tensor counter, at::Tensor overflow, double eps)
{
    gen_candidates_overlap_cuda(aabb_min, aabb_max, tri_min, tri_max,
                                a_offset, t_offset, cand_a_out, cand_t_out, counter, overflow, eps);
}

std::vector<at::Tensor> aabb_tri_sat_clip_select(
    at::Tensor aabbs_min, at::Tensor aabbs_max,
    at::Tensor tris_verts,
    at::Tensor cand_a_idx, at::Tensor cand_t_idx,
    int64_t mode, double eps, int64_t max_vert)
{
    return aabb_tri_sat_clip_select_cuda(aabbs_min, aabbs_max, tris_verts,
                                         cand_a_idx, cand_t_idx, mode, eps, max_vert);
}

void voxelize_mark(
    at::Tensor aabb_min, at::Tensor aabb_max,
    at::Tensor tri_min,  at::Tensor tri_max,
    at::Tensor tris_verts,
    int64_t a_offset, int64_t t_offset,
    at::Tensor active_mask,
    bool use_sat, double eps)
{
    voxelize_mark_cuda(aabb_min, aabb_max, tri_min, tri_max, tris_verts,
                       a_offset, t_offset, active_mask, use_sat, eps);
}


// ★ 最终版：高性能融合内核的绑定
std::vector<at::Tensor> segment_tri_intersection_fused(
    at::Tensor seg_verts,
    at::Tensor tris_verts,
    at::Tensor tri_aabb_min,
    at::Tensor tri_aabb_max,
    double eps)
{
    return segment_tri_intersection_fused_cuda(
        seg_verts, tris_verts, tri_aabb_min, tri_aabb_max, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gen_candidates_overlap", &gen_candidates_overlap, "Broad-phase candidates by bbox overlap (CUDA)");
  m.def("aabb_tri_sat_clip_select", &aabb_tri_sat_clip_select, "Narrow-phase selectable: 0=hits, 1=centroid+area, 2=full polys");
  m.def("voxelize_mark", &voxelize_mark, "Fast voxel AABB marking (CUDA) with optional SAT");
  m.def("segment_tri_intersection_fused", &segment_tri_intersection_fused, "Fused segment-triangle intersection (CUDA)");
}