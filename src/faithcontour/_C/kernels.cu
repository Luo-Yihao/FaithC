#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

template<typename T> 
__device__ __forceinline__ T absf(T x) { 
    return fabs(x);
}

template<typename T>
__device__ __forceinline__ int argmax3abs(T x, T y, T z) {
    T ax = fabs(x);
    T ay = fabs(y);
    T az = fabs(z);
    return (ax >= ay && ax >= az) ? 0 : ((ay >= az) ? 1 : 2);
}

// 将3D点投影到去掉主法向分量后的2D平面
template<typename T>
__device__ __forceinline__ void project_to_2d(int drop, T x, T y, T z, T& u, T& v){
    if (drop==0){ u = y; v = z; }   // drop X -> use (y,z)
    else if (drop==1){ u = x; v = z;} // drop Y -> use (x,z)
    else { u = x; v = y; }          // drop Z -> use (x,y)
}

template<typename T>
__device__ __forceinline__ T orient2d(T ax,T ay,T bx,T by,T cx,T cy){
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
}



const int FUSED_KERNEL_TILE_SIZE = 256;

__global__ void k_segment_tri_intersection_fused_float(
    const float* __restrict__ seg_verts,
    const float* __restrict__ tris_verts,
    const float* __restrict__ tri_aabb_min,
    const float* __restrict__ tri_aabb_max,
    int64_t num_segs,
    int64_t num_tris,
    float eps,
    long* __restrict__ out_seg_indices,
    long* __restrict__ out_tri_indices,
    float* __restrict__ out_dots,
    int* __restrict__ counter)
{
    int seg_idx = blockIdx.x;
    if (seg_idx >= num_segs) return;

    const float* seg_ptr = seg_verts + seg_idx * 6;
    float p0x = seg_ptr[0], p0y = seg_ptr[1], p0z = seg_ptr[2];
    float p1x = seg_ptr[3], p1y = seg_ptr[4], p1z = seg_ptr[5];

    float seg_min_x = fminf(p0x, p1x); float seg_min_y = fminf(p0y, p1y); float seg_min_z = fminf(p0z, p1z);
    float seg_max_x = fmaxf(p0x, p1x); float seg_max_y = fmaxf(p0y, p1y); float seg_max_z = fmaxf(p0z, p1z);
    float seg_dir_x = p1x - p0x; float seg_dir_y = p1y - p0y; float seg_dir_z = p1z - p0z;

    extern __shared__ float smem_fused_kernel_float[];
    float* tri_min_tile = smem_fused_kernel_float;
    float* tri_max_tile = tri_min_tile + FUSED_KERNEL_TILE_SIZE * 3;

    for (int tile_start = 0; tile_start < num_tris; tile_start += FUSED_KERNEL_TILE_SIZE) {
        
        int local_idx = threadIdx.x;
        int global_tri_idx_to_load = tile_start + local_idx;

        if (global_tri_idx_to_load < num_tris) {
            const float* min_ptr = tri_aabb_min + global_tri_idx_to_load * 3;
            const float* max_ptr = tri_aabb_max + global_tri_idx_to_load * 3;
            tri_min_tile[local_idx * 3 + 0] = min_ptr[0];
            tri_min_tile[local_idx * 3 + 1] = min_ptr[1];
            tri_min_tile[local_idx * 3 + 2] = min_ptr[2];
            tri_max_tile[local_idx * 3 + 0] = max_ptr[0];
            tri_max_tile[local_idx * 3 + 1] = max_ptr[1];
            tri_max_tile[local_idx * 3 + 2] = max_ptr[2];
        }
        __syncthreads();

        int current_global_tri_idx = tile_start + threadIdx.x;
        if (current_global_tri_idx < num_tris) {
            
            bool overlap = (seg_max_x >= tri_min_tile[threadIdx.x*3+0] && seg_min_x <= tri_max_tile[threadIdx.x*3+0]) &&
                           (seg_max_y >= tri_min_tile[threadIdx.x*3+1] && seg_min_y <= tri_max_tile[threadIdx.x*3+1]) &&
                           (seg_max_z >= tri_min_tile[threadIdx.x*3+2] && seg_min_z <= tri_max_tile[threadIdx.x*3+2]);

            if (overlap) {
                const float* tri_ptr = tris_verts + current_global_tri_idx * 9;
                float v0x = tri_ptr[0], v0y = tri_ptr[1], v0z = tri_ptr[2];
                float v1x = tri_ptr[3], v1y = tri_ptr[4], v1z = tri_ptr[5];
                float v2x = tri_ptr[6], v2y = tri_ptr[7], v2z = tri_ptr[8];

                float edge1_x = v1x-v0x, edge1_y = v1y-v0y, edge1_z = v1z-v0z;
                float edge2_x = v2x-v0x, edge2_y = v2y-v0y, edge2_z = v2z-v0z;

                float pvec_x = seg_dir_y * edge2_z - seg_dir_z * edge2_y;
                float pvec_y = seg_dir_z * edge2_x - seg_dir_x * edge2_z;
                float pvec_z = seg_dir_x * edge2_y - seg_dir_y * edge2_x;
                float det = edge1_x * pvec_x + edge1_y * pvec_y + edge1_z * pvec_z;
                
                if (! (absf(det) < eps) ) {
                    float tvec_x = p0x - v0x, tvec_y = p0y - v0y, tvec_z = p0z - v0z;
                    float qvec_x = tvec_y * edge1_z - tvec_z * edge1_y;
                    float qvec_y = tvec_z * edge1_x - tvec_x * edge1_z;
                    float qvec_z = tvec_x * edge1_y - tvec_y * edge1_x;

                    float u = tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z;
                    float v = seg_dir_x * qvec_x + seg_dir_y * qvec_y + seg_dir_z * qvec_z;
                    float t = edge2_x * qvec_x + edge2_y * qvec_y + edge2_z * qvec_z;

                    bool hit_pos = (det > eps) && (u>=0.0f) && (u<=det) && (v>=0.0f) && (u+v<=det) && (t>=0.0f) && (t<=det);
                    bool hit_neg = (det < -eps) && (u<=0.0f) && (u>=det) && (v<=0.0f) && (u+v>=det) && (t<=0.0f) && (t>=det);

                    if (hit_pos || hit_neg) {
                        int write_idx = atomicAdd(counter, 1);
                        out_seg_indices[write_idx] = seg_idx;
                        out_tri_indices[write_idx] = current_global_tri_idx;

                        float nx = edge1_y*edge2_z - edge1_z*edge2_y;
                        float ny = edge1_z*edge2_x - edge1_x*edge2_z;
                        float nz = edge1_x*edge2_y - edge1_y*edge2_x;
                        float norm_sq = nx*nx + ny*ny + nz*nz;

                        if (norm_sq > eps) {
                            float inv_norm = rsqrtf(norm_sq);
                            out_dots[write_idx] = (seg_dir_x*nx + seg_dir_y*ny + seg_dir_z*nz) * inv_norm;
                        } else {
                            out_dots[write_idx] = 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}


std::vector<at::Tensor> segment_tri_intersection_fused_cuda(
    at::Tensor seg_verts,
    at::Tensor tris_verts,
    at::Tensor tri_aabb_min,
    at::Tensor tri_aabb_max,
    double eps)
{
    TORCH_CHECK(seg_verts.scalar_type() == torch::kFloat32, "Fused intersection kernel currently only supports float32 tensors.");
    
    const auto num_segs = seg_verts.size(0);
    const auto num_tris = tris_verts.size(0);
    const int64_t max_hits_guess = num_segs * 8 + 4096;

    auto opts_l = seg_verts.options().dtype(torch::kInt64);
    auto opts_f = seg_verts.options();
    auto opts_i = seg_verts.options().dtype(torch::kInt32);

    auto out_seg_indices = torch::empty({max_hits_guess}, opts_l);
    auto out_tri_indices = torch::empty({max_hits_guess}, opts_l);
    auto out_dots        = torch::empty({max_hits_guess}, opts_f);
    auto counter         = torch::zeros({1}, opts_i);

    dim3 blocks(num_segs);
    const int threads = FUSED_KERNEL_TILE_SIZE;
    size_t shared_mem_size = threads * 3 * 2 * sizeof(float);

    k_segment_tri_intersection_fused_float<<<blocks, threads, shared_mem_size>>>(
        seg_verts.data_ptr<float>(), tris_verts.data_ptr<float>(),
        tri_aabb_min.data_ptr<float>(), tri_aabb_max.data_ptr<float>(),
        num_segs, num_tris, static_cast<float>(eps),
        out_seg_indices.data_ptr<long>(), out_tri_indices.data_ptr<long>(),
        out_dots.data_ptr<float>(), counter.data_ptr<int>());

    int final_hit_count = counter.item<int>();
    TORCH_CHECK(final_hit_count <= max_hits_guess, "Intersection count exceeded pre-allocated buffer.");
    
    return {
        out_seg_indices.slice(0, 0, final_hit_count),
        out_tri_indices.slice(0, 0, final_hit_count),
        out_dots.slice(0, 0, final_hit_count)
    };
}


// ===================================================================
// ★ 恢复的原有函数CUDA实现 ★
// ===================================================================


template<typename T>
__device__ __forceinline__ bool on_seg_2d(T ax,T ay,T bx,T by,T px,T py, T eps){
    T minx = ax < bx ? ax : bx, maxx = ax > bx ? ax : bx;
    T miny = ay < by ? ay : by, maxy = ay > by ? ay : by;
    if (px < minx - eps || px > maxx + eps || py < miny - eps || py > maxy + eps) return false;
    T area = orient2d(ax,ay,bx,by,px,py);
    return fabs((double)area) <= (double)eps;
}

template<typename T>
__device__ __forceinline__ bool seg_seg_intersect_2d(T ax,T ay,T bx,T by,
                                                     T cx,T cy,T dx,T dy, T eps){
    T minabx = ax<bx?ax:bx, maxabx = ax>bx?ax:bx;
    T minaby = ay<by?ay:by, maxaby = ay>by?ay:by;
    T mincdx = cx<dx?cx:dx, maxcdx = cx>dx?cx:dx;
    T mincdy = cy<dy?cy:dy, maxcdy = cy>dy?cy:dy;
    if (maxabx < mincdx - eps || maxcdx < minabx - eps ||
        maxaby < mincdy - eps || maxcdy < minaby - eps) return false;

    T o1 = orient2d(ax,ay,bx,by,cx,cy);
    T o2 = orient2d(ax,ay,bx,by,dx,dy);
    T o3 = orient2d(cx,cy,dx,dy,ax,ay);
    T o4 = orient2d(cx,cy,dx,dy,bx,by);

    if ((o1>eps && o2<-eps || o1<-eps && o2>eps) &&
        (o3>eps && o4<-eps || o3<-eps && o4>eps)) return true;

    if (fabs((double)o1) <= (double)eps && on_seg_2d(ax,ay,bx,by,cx,cy,eps)) return true;
    if (fabs((double)o2) <= (double)eps && on_seg_2d(ax,ay,bx,by,dx,dy,eps)) return true;
    if (fabs((double)o3) <= (double)eps && on_seg_2d(cx,cy,dx,dy,ax,ay,eps)) return true;
    if (fabs((double)o4) <= (double)eps && on_seg_2d(cx,cy,dx,dy,bx,by,eps)) return true;

    return false;
}

template<typename T>
__device__ __forceinline__ bool point_in_tri_2d(T px,T py,
                                                T ax,T ay,T bx,T by,T cx,T cy,
                                                T eps){
    T o0 = orient2d(ax,ay,bx,by,px,py);
    T o1 = orient2d(bx,by,cx,cy,px,py);
    T o2 = orient2d(cx,cy,ax,ay,px,py);
    bool all_pos = (o0 >= -eps) && (o1 >= -eps) && (o2 >= -eps);
    bool all_neg = (o0 <=  eps) && (o1 <=  eps) && (o2 <=  eps);
    return all_pos || all_neg;
}


// --------------------------------------
// Preprocess triangles: v0, e1, e2, normal, inv_norm
// --------------------------------------
template<typename scalar_t>
__global__ void k_preprocess_tris(
    const scalar_t* __restrict__ tris, int64_t M,
    scalar_t* __restrict__ v0, scalar_t* __restrict__ e1, scalar_t* __restrict__ e2,
    scalar_t* __restrict__ nrm, scalar_t* __restrict__ invn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    const scalar_t* t = tris + i*9;
    scalar_t v0x = t[0], v0y = t[1], v0z = t[2];
    scalar_t v1x = t[3], v1y = t[4], v1z = t[5];
    scalar_t v2x = t[6], v2y = t[7], v2z = t[8];
    scalar_t e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    scalar_t e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    scalar_t nx = e1y*e2z - e1z*e2y;
    scalar_t ny = e1z*e2x - e1x*e2z;
    scalar_t nz = e1x*e2y - e1y*e2x;
    scalar_t nn = nx*nx + ny*ny + nz*nz;
    invn[i] = rsqrt(fmax(nn, static_cast<scalar_t>(1e-38)));
    v0[i*3+0]=v0x; v0[i*3+1]=v0y; v0[i*3+2]=v0z;
    e1[i*3+0]=e1x; e1[i*3+1]=e1y; e1[i*3+2]=e1z;
    e2[i*3+0]=e2x; e2[i*3+1]=e2y; e2[i*3+2]=e2z;
    nrm[i*3+0]=nx;  nrm[i*3+1]=ny;  nrm[i*3+2]=nz;
}

template<typename scalar_t>
void preprocess_tris_cuda(
    at::Tensor tris_verts, at::Tensor tri_v0, at::Tensor tri_e1,
    at::Tensor tri_e2, at::Tensor tri_normal, at::Tensor tri_inv_norm)
{
    const auto M = tris_verts.size(0);
    if (M == 0) return;
    k_preprocess_tris<scalar_t><<<(M + 255) / 256, 256>>>(
        tris_verts.data_ptr<scalar_t>(), M,
        tri_v0.data_ptr<scalar_t>(), tri_e1.data_ptr<scalar_t>(),
        tri_e2.data_ptr<scalar_t>(), tri_normal.data_ptr<scalar_t>(),
        tri_inv_norm.data_ptr<scalar_t>());
}

// --------------------------------------
// Broadphase AABB overlap
// --------------------------------------
template<typename T>
__global__ void k_gen_candidates_overlap(
    const T* __restrict__ aabb_min, const T* __restrict__ aabb_max,
    const T* __restrict__ tri_min,  const T* __restrict__ tri_max,
    int64_t Na, int64_t Nt,
    int64_t a_offset, int64_t t_offset,
    long* __restrict__ cand_a_out, long* __restrict__ cand_t_out,
    int* __restrict__ counter, unsigned char* __restrict__ overflow, long cap, T eps)
{
    int ai = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y * blockDim.y + threadIdx.y;
    if (ai>=Na || ti>=Nt) return;

    const T* amin = aabb_min + ai*3;
    const T* amax = aabb_max + ai*3;
    const T* tmin = tri_min  + ti*3;
    const T* tmax = tri_max  + ti*3;

    bool ov = (amin[0] <= tmax[0] + eps && amax[0] >= tmin[0] - eps) &&
          (amin[1] <= tmax[1] + eps && amax[1] >= tmin[1] - eps) &&
          (amin[2] <= tmax[2] + eps && amax[2] >= tmin[2] - eps);
    if (!ov) return;

    int idx = atomicAdd(counter, 1);
    if ((long)idx >= cap){ *overflow = 1; return; }
    cand_a_out[idx] = (long)(a_offset + ai);
    cand_t_out[idx] = (long)(t_offset + ti);
}

void gen_candidates_overlap_cuda(
    at::Tensor aabb_min, at::Tensor aabb_max,
    at::Tensor tri_min,  at::Tensor tri_max,
    int64_t a_offset, int64_t t_offset,
    at::Tensor cand_a_out, at::Tensor cand_t_out,
    at::Tensor counter, at::Tensor overflow, double eps)
{
    const auto Na = aabb_min.size(0);
    const auto Nt = tri_min.size(0);
    dim3 blk(32,32);
    dim3 grd((Na+blk.x-1)/blk.x, (Nt+blk.y-1)/blk.y);

    AT_DISPATCH_FLOATING_TYPES(aabb_min.scalar_type(), "k_gen_candidates_overlap", [&]{
        k_gen_candidates_overlap<scalar_t><<<grd, blk>>>(
            aabb_min.data_ptr<scalar_t>(), aabb_max.data_ptr<scalar_t>(),
            tri_min.data_ptr<scalar_t>(),  tri_max.data_ptr<scalar_t>(),
            Na, Nt, a_offset, t_offset,
            cand_a_out.data_ptr<long>(), cand_t_out.data_ptr<long>(),
            counter.data_ptr<int>(), overflow.data_ptr<unsigned char>(),
            (long)cand_a_out.size(0),
            (scalar_t)eps );
    });

}

// --------------------------------------
// SAT/Clip（原有实现）
// --------------------------------------
template<typename T>
__device__ __forceinline__ bool proj_overlap_on_axis(
    const T v[3][3], const T axis[3], const T he[3], T eps) {
    T axis_len2 = axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2];
    if (axis_len2 < T(1e-20)) return true;
    T p0 = v[0][0]*axis[0] + v[0][1]*axis[1] + v[0][2]*axis[2];
    T p1 = v[1][0]*axis[0] + v[1][1]*axis[1] + v[1][2]*axis[2];
    T p2 = v[2][0]*axis[0] + v[2][1]*axis[1] + v[2][2]*axis[2];
    T tri_min = min(p0, min(p1, p2));
    T tri_max = max(p0, max(p1, p2));
    T r = absf(axis[0])*he[0] + absf(axis[1])*he[1] + absf(axis[2])*he[2];
    if (tri_min >  r + eps) return false;
    if (tri_max < -r - eps) return false;
    return true;
}

template<typename T>
__device__ __forceinline__ bool tri_aabb_sat(
    const T center[3], const T he[3], const T tri[3][3], T eps) {
    T v[3][3];
    #pragma unroll
    for(int i=0;i<3;++i){ v[i][0] = tri[i][0]-center[0]; v[i][1] = tri[i][1]-center[1]; v[i][2] = tri[i][2]-center[2]; }
    for (int ax=0; ax<3; ++ax){
        T mn=v[0][ax], mx=v[0][ax];
        mn = min(mn, v[1][ax]); mx = max(mx, v[1][ax]); mn = min(mn, v[2][ax]); mx = max(mx, v[2][ax]);
        if (mn > he[ax] + eps) return false;
        if (mx < -he[ax] - eps) return false;
    }
    T e0[3]={v[1][0]-v[0][0], v[1][1]-v[0][1], v[1][2]-v[0][2]};
    T e1[3]={v[2][0]-v[1][0], v[2][1]-v[1][1], v[2][2]-v[1][2]};
    T n[3] ={e0[1]*e1[2]-e0[2]*e1[1], e0[2]*e1[0]-e0[0]*e1[2], e0[0]*e1[1]-e0[1]*e1[0]};
    T nlen = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    if (nlen > T(1e-20)){
        T p0 = (v[0][0]*n[0] + v[0][1]*n[1] + v[0][2]*n[2]) / nlen;
        T r  = (absf(n[0])*he[0] + absf(n[1])*he[1] + absf(n[2])*he[2]) / nlen;
        if (absf(p0) > r + eps) return false;
    }
    T e2[3]={v[0][0]-v[2][0], v[0][1]-v[2][1], v[0][2]-v[2][2]};
    const T X[3]={1,0,0}, Y[3]={0,1,0}, Z[3]={0,0,1};
    T a[3];
    #define TEST_AXIS(ax, ey) a[0]=ax[1]*ey[2]-ax[2]*ey[1]; a[1]=ax[2]*ey[0]-ax[0]*ey[2]; a[2]=ax[0]*ey[1]-ax[1]*ey[0]; if(!proj_overlap_on_axis(v,a,he,eps)) return false;
    TEST_AXIS(X, e0); TEST_AXIS(Y, e0); TEST_AXIS(Z, e0);
    TEST_AXIS(X, e1); TEST_AXIS(Y, e1); TEST_AXIS(Z, e1);
    TEST_AXIS(X, e2); TEST_AXIS(Y, e2); TEST_AXIS(Z, e2);
    #undef TEST_AXIS
    return true;
}

template<typename T, int MAXV>
__device__ __forceinline__ int clip_with_plane(const T in_poly[MAXV][3], int in_count, const T n[3], T d, T out_poly[MAXV][3], T eps) {
    if (in_count<=0) return 0;
    T dist[MAXV]; bool inside[MAXV];
    #pragma unroll
    for(int i=0;i<in_count;++i){
        dist[i] = in_poly[i][0]*n[0] + in_poly[i][1]*n[1] + in_poly[i][2]*n[2] - d;
        inside[i] = (dist[i] <= eps);
    }
    int out_count=0;
    #pragma unroll
    for(int i=0;i<in_count;++i){
        int j=(i+1==in_count)?0:(i+1);
        const T* P=in_poly[i]; const T* Q=in_poly[j];
        T dP=dist[i], dQ=dist[j];
        bool inP=inside[i], inQ=inside[j];
        if (inP && inQ){
            out_poly[out_count][0]=Q[0]; out_poly[out_count][1]=Q[1]; out_poly[out_count][2]=Q[2];
            ++out_count;
        }else if (inP && !inQ){
            T denom=dP-dQ; if (::fabs((double)denom)>eps){
                T t=dP/denom;
                out_poly[out_count][0]=P[0]+t*(Q[0]-P[0]);
                out_poly[out_count][1]=P[1]+t*(Q[1]-P[1]);
                out_poly[out_count][2]=P[2]+t*(Q[2]-P[2]);
                ++out_count;
            }
        }else if (!inP && inQ){
            T denom=dP-dQ; if (::fabs((double)denom)>eps){
                T t=dP/denom;
                out_poly[out_count][0]=P[0]+t*(Q[0]-P[0]);
                out_poly[out_count][1]=P[1]+t*(Q[1]-P[1]);
                out_poly[out_count][2]=P[2]+t*(Q[2]-P[2]);
                ++out_count;
            }
            out_poly[out_count][0]=Q[0]; out_poly[out_count][1]=Q[1]; out_poly[out_count][2]=Q[2];
            ++out_count;
        }
        if (out_count>=MAXV) return MAXV;
    }
    return out_count;
}

// Narrowphase Kernels for aabb_tri_sat_clip_select
template<typename T>
__global__ void sat_hit_kernel(const T* __restrict__ aabbs_min, const T* __restrict__ aabbs_max, const T* __restrict__ tris_verts, const long* __restrict__ cand_a, const long* __restrict__ cand_t, int64_t K, T eps, bool* __restrict__ hit_mask, long* __restrict__ out_a_idx, long* __restrict__ out_t_idx) {
    int k=blockIdx.x*blockDim.x+threadIdx.x; if (k>=K) return;
    long ai=cand_a[k], ti=cand_t[k];
    const T* bmin=aabbs_min+ai*3; const T* bmax=aabbs_max+ai*3;
    T center[3]={(bmin[0]+bmax[0])*T(0.5),(bmin[1]+bmax[1])*T(0.5),(bmin[2]+bmax[2])*T(0.5)};
    T he[3]={(bmax[0]-bmin[0])*T(0.5),(bmax[1]-bmin[1])*T(0.5),(bmax[2]-bmin[2])*T(0.5)};
    T tri[3][3];
    #pragma unroll
    for(int v=0;v<3;++v){ const T* tv=tris_verts+(ti*3+v)*3; tri[v][0]=tv[0]; tri[v][1]=tv[1]; tri[v][2]=tv[2]; }
    bool inter=tri_aabb_sat<T>(center,he,tri,eps);
    hit_mask[k]=inter; out_a_idx[k]=ai; out_t_idx[k]=ti;
}

template<typename T, int MAXV>
__global__ void sat_centroid_kernel(
    const T* __restrict__ aabbs_min,
    const T* __restrict__ aabbs_max,
    const T* __restrict__ tris_verts,
    const long* __restrict__ cand_a,
    const long* __restrict__ cand_t,
    int64_t K, T eps,
    bool* __restrict__ hit_mask,
    int* __restrict__ poly_counts,
    T* __restrict__ centroids,
    T* __restrict__ areas,
    long* __restrict__ out_a_idx,
    long* __restrict__ out_t_idx)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const long ai = cand_a[k];
    const long ti = cand_t[k];
    out_a_idx[k] = ai;
    out_t_idx[k] = ti;

    const T* bmin = aabbs_min + ai*3;
    const T* bmax = aabbs_max + ai*3;

    // 三角形
    T tri[3][3];
    #pragma unroll
    for (int v=0; v<3; ++v){
        const T* tv = tris_verts + (ti*3 + v)*3;
        tri[v][0]=tv[0]; tri[v][1]=tv[1]; tri[v][2]=tv[2];
    }

    // 初始多边形 = 三角形
    T polyA[MAXV][3], polyB[MAXV][3];
    int nA = 3;
    for (int i=0;i<3;++i){
        polyA[i][0]=tri[i][0];
        polyA[i][1]=tri[i][1];
        polyA[i][2]=tri[i][2];
    }

    // clip 六个平面
    const T nxp[3]={T(1),0,0},   nyp[3]={0,T(1),0},   nzp[3]={0,0,T(1)};
    const T nxn[3]={T(-1),0,0},  nyn[3]={0,T(-1),0},  nzn[3]={0,0,T(-1)};
    auto clip_once = [&](const T inP[MAXV][3], int inN, const T n[3], T d, T outP[MAXV][3])->int{
        return clip_with_plane<T,MAXV>(inP,inN,n,d,outP,eps);
    };

    int nB = clip_once(polyA,nA,nxp,bmax[0],polyB); nA=nB;
    if(!nA){ poly_counts[k]=0; areas[k]=T(0); centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return; }
    nB = clip_once(polyB,nA,nxn,-bmin[0],polyA); nA=nB;
    if(!nA){ poly_counts[k]=0; areas[k]=T(0); centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return; }
    nB = clip_once(polyA,nA,nyp,bmax[1],polyB); nA=nB;
    if(!nA){ poly_counts[k]=0; areas[k]=T(0); centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return; }
    nB = clip_once(polyB,nA,nyn,-bmin[1],polyA); nA=nB;
    if(!nA){ poly_counts[k]=0; areas[k]=T(0); centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return; }
    nB = clip_once(polyA,nA,nzp,bmax[2],polyB); nA=nB;
    if(!nA){ poly_counts[k]=0; areas[k]=T(0); centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return; }
    nA = clip_once(polyB,nB,nzn,-bmin[2],polyA);
    if(!nA){ poly_counts[k]=0; areas[k]=T(0); centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return; }

    int cnt = nA; if (cnt>MAXV) cnt=MAXV;
    poly_counts[k]=cnt;
    if(cnt<1){
        areas[k]=0; centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return;
    }

    // --- 算术平均 ---
    T cx=0, cy=0, cz=0;
    for(int i=0;i<cnt;++i){ cx+=polyA[i][0]; cy+=polyA[i][1]; cz+=polyA[i][2]; }
    cx/=T(cnt); cy/=T(cnt); cz/=T(cnt);

    // clamp 到 voxel
    cx = fminf(fmaxf(cx,bmin[0]),bmax[0]);
    cy = fminf(fmaxf(cy,bmin[1]),bmax[1]);
    cz = fminf(fmaxf(cz,bmin[2]),bmax[2]);

    // 投影到三角形平面并 barycentric clamp
    T ax=tri[0][0], ay=tri[0][1], az=tri[0][2];
    T bx=tri[1][0], by=tri[1][1], bz=tri[1][2];
    T cx_t=tri[2][0], cy_t=tri[2][1], cz_t=tri[2][2];
    T e1x=bx-ax, e1y=by-ay, e1z=bz-az;
    T e2x=cx_t-ax, e2y=cy_t-ay, e2z=cz_t-az;
    T vx=cx-ax, vy=cy-ay, vz=cz-az;

    T d00=e1x*e1x+e1y*e1y+e1z*e1z;
    T d01=e1x*e2x+e1y*e2y+e1z*e2z;
    T d11=e2x*e2x+e2y*e2y+e2z*e2z;
    T d20=vx*e1x+vy*e1y+vz*e1z;
    T d21=vx*e2x+vy*e2y+vz*e2z;
    T denom=d00*d11-d01*d01;
    T v_bc=(d11*d20-d01*d21)/denom;
    T w_bc=(d00*d21-d01*d20)/denom;
    T u_bc=1.0f-v_bc-w_bc;

    // clamp bary
    if(u_bc<0) u_bc=0;
    if(v_bc<0) v_bc=0;
    if(w_bc<0) w_bc=0;
    T norm=u_bc+v_bc+w_bc; if(norm<=eps){u_bc=1; v_bc=0; w_bc=0; norm=1;}
    u_bc/=norm; v_bc/=norm; w_bc/=norm;

    T projx=u_bc*ax+v_bc*bx+w_bc*cx_t;
    T projy=u_bc*ay+v_bc*by+w_bc*cy_t;
    T projz=u_bc*az+v_bc*bz+w_bc*cz_t;

    centroids[k*3+0]=projx;
    centroids[k*3+1]=projy;
    centroids[k*3+2]=projz;
    areas[k]=T(1.0); // 这里你如果不关心面积，可以统一写 1
    hit_mask[k]=true;
}



template<typename T, int MAXV>
__global__ void sat_clip_kernel(
    const T* __restrict__ aabbs_min, const T* __restrict__ aabbs_max,
    const T* __restrict__ tris_verts,
    const long* __restrict__ cand_a, const long* __restrict__ cand_t,
    int64_t K, T eps,
    bool* __restrict__ hit_mask,
    int* __restrict__ poly_counts,
    T* __restrict__ poly_verts,
    T* __restrict__ centroids,
    T* __restrict__ areas,
    long* __restrict__ out_a_idx,
    long* __restrict__ out_t_idx)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const long ai = cand_a[k];
    const long ti = cand_t[k];

    out_a_idx[k] = ai;
    out_t_idx[k] = ti;

    const T* bmin = aabbs_min + ai*3;
    const T* bmax = aabbs_max + ai*3;

    hit_mask[k] = true;

    // 三角形
    T tri[3][3];
    #pragma unroll
    for (int v=0; v<3; ++v){
        const T* tv = tris_verts + (ti*3 + v)*3;
        tri[v][0]=tv[0]; tri[v][1]=tv[1]; tri[v][2]=tv[2];
    }

    // 初始 polygon = 三角形
    T polyA[MAXV][3], polyB[MAXV][3];
    int nA=3;
    for (int i=0;i<3;++i){
        polyA[i][0]=tri[i][0];
        polyA[i][1]=tri[i][1];
        polyA[i][2]=tri[i][2];
    }

    // clip 六个面
    const T nxp[3]={T(1),0,0},   nyp[3]={0,T(1),0},   nzp[3]={0,0,T(1)};
    const T nxn[3]={T(-1),0,0},  nyn[3]={0,T(-1),0},  nzn[3]={0,0,T(-1)};
    auto clip_once = [&](const T inP[MAXV][3], int inN, const T n[3], T d, T outP[MAXV][3])->int{
        return clip_with_plane<T,MAXV>(inP,inN,n,d,outP,eps);
    };

    int nB=clip_once(polyA,nA,nxp,bmax[0],polyB); nA=nB;
    if(!nA){poly_counts[k]=0;areas[k]=0;centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0;return;}
    nB=clip_once(polyB,nA,nxn,-bmin[0],polyA); nA=nB;
    if(!nA){poly_counts[k]=0;areas[k]=0;centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0;return;}
    nB=clip_once(polyA,nA,nyp,bmax[1],polyB); nA=nB;
    if(!nA){poly_counts[k]=0;areas[k]=0;centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0;return;}
    nB=clip_once(polyB,nA,nyn,-bmin[1],polyA); nA=nB;
    if(!nA){poly_counts[k]=0;areas[k]=0;centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0;return;}
    nB=clip_once(polyA,nA,nzp,bmax[2],polyB); nA=nB;
    if(!nA){poly_counts[k]=0;areas[k]=0;centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0;return;}
    nA=clip_once(polyB,nB,nzn,-bmin[2],polyA);
    if(!nA){poly_counts[k]=0;areas[k]=0;centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0;return;}

    int cnt=nA; if(cnt>MAXV) cnt=MAXV;
    poly_counts[k]=cnt;

    // 清空
    for(int i=0;i<MAXV;++i){
        poly_verts[(k*MAXV+i)*3+0]=0;
        poly_verts[(k*MAXV+i)*3+1]=0;
        poly_verts[(k*MAXV+i)*3+2]=0;
    }
    for(int i=0;i<cnt;++i){
        poly_verts[(k*MAXV+i)*3+0]=polyA[i][0];
        poly_verts[(k*MAXV+i)*3+1]=polyA[i][1];
        poly_verts[(k*MAXV+i)*3+2]=polyA[i][2];
    }

    if(cnt<1){
        areas[k]=0; centroids[k*3+0]=centroids[k*3+1]=centroids[k*3+2]=0; return;
    }

    // --- 算术平均 ---
    T cx=0, cy=0, cz=0;
    for(int i=0;i<cnt;++i){ cx+=polyA[i][0]; cy+=polyA[i][1]; cz+=polyA[i][2]; }
    cx/=T(cnt); cy/=T(cnt); cz/=T(cnt);

    // clamp 到 voxel
    cx=fminf(fmaxf(cx,bmin[0]),bmax[0]);
    cy=fminf(fmaxf(cy,bmin[1]),bmax[1]);
    cz=fminf(fmaxf(cz,bmin[2]),bmax[2]);

    // 投影到三角面片
    T ax=tri[0][0], ay=tri[0][1], az=tri[0][2];
    T bx=tri[1][0], by=tri[1][1], bz=tri[1][2];
    T cx_t=tri[2][0], cy_t=tri[2][1], cz_t=tri[2][2];
    T e1x=bx-ax, e1y=by-ay, e1z=bz-az;
    T e2x=cx_t-ax, e2y=cy_t-ay, e2z=cz_t-az;
    T vx=cx-ax, vy=cy-ay, vz=cz-az;

    T d00=e1x*e1x+e1y*e1y+e1z*e1z;
    T d01=e1x*e2x+e1y*e2y+e1z*e2z;
    T d11=e2x*e2x+e2y*e2y+e2z*e2z;
    T d20=vx*e1x+vy*e1y+vz*e1z;
    T d21=vx*e2x+vy*e2y+vz*e2z;
    T denom=d00*d11-d01*d01;
    T v_bc=(d11*d20-d01*d21)/denom;
    T w_bc=(d00*d21-d01*d20)/denom;
    T u_bc=1.0f-v_bc-w_bc;

    if(u_bc<0) u_bc=0;
    if(v_bc<0) v_bc=0;
    if(w_bc<0) w_bc=0;
    T norm=u_bc+v_bc+w_bc; if(norm<=eps){u_bc=1;v_bc=0;w_bc=0;norm=1;}
    u_bc/=norm; v_bc/=norm; w_bc/=norm;

    T projx=u_bc*ax+v_bc*bx+w_bc*cx_t;
    T projy=u_bc*ay+v_bc*by+w_bc*cy_t;
    T projz=u_bc*az+v_bc*bz+w_bc*cz_t;

    centroids[k*3+0]=projx;
    centroids[k*3+1]=projy;
    centroids[k*3+2]=projz;
    areas[k]=T(1.0);
}

std::vector<at::Tensor> aabb_tri_sat_clip_select_cuda(
    at::Tensor aabbs_min, at::Tensor aabbs_max,
    at::Tensor tris_verts,
    at::Tensor cand_a_idx, at::Tensor cand_t_idx,
    int64_t mode, double eps, int64_t max_vert)
{
    const auto K = cand_a_idx.size(0);
    auto opts_b = aabbs_min.options().dtype(torch::kBool);
    auto opts_i = aabbs_min.options().dtype(torch::kInt32);
    auto opts_l = aabbs_min.options().dtype(torch::kInt64);
    auto opts_f = aabbs_min.options();

    auto hit_mask  = torch::empty({K}, opts_b);
    auto out_a_idx = torch::empty({K}, opts_l);
    auto out_t_idx = torch::empty({K}, opts_l);

    at::Tensor poly_counts, poly_verts, centroids, areas;

    const int threads = 256;
    const int blocks=(K+threads-1)/threads;

    if (mode==0){
        AT_DISPATCH_FLOATING_TYPES(aabbs_min.scalar_type(), "sat_hit_kernel", [&]{
            sat_hit_kernel<scalar_t><<<blocks,threads>>>(
                aabbs_min.data_ptr<scalar_t>(), aabbs_max.data_ptr<scalar_t>(),
                tris_verts.data_ptr<scalar_t>(),
                cand_a_idx.data_ptr<long>(), cand_t_idx.data_ptr<long>(),
                K, (scalar_t)eps, hit_mask.data_ptr<bool>(),
                out_a_idx.data_ptr<long>(), out_t_idx.data_ptr<long>());
        });
        poly_counts=torch::empty({0}, opts_i);
        poly_verts =torch::empty({0,0,3}, opts_f);
        centroids  =torch::empty({0,3}, opts_f);
        areas      =torch::empty({0}, opts_f);
    } else if (mode == 1) {
        poly_counts = torch::zeros({K}, opts_i);
        centroids  =torch::zeros({K,3}, opts_f);
        areas      =torch::zeros({K},   opts_f);
        AT_DISPATCH_FLOATING_TYPES(aabbs_min.scalar_type(), "sat_centroid_kernel", [&]{
            (max_vert==8 ? sat_centroid_kernel<scalar_t,8> : sat_centroid_kernel<scalar_t,7>)
                <<<blocks,threads>>>(
                    aabbs_min.data_ptr<scalar_t>(), aabbs_max.data_ptr<scalar_t>(),
                    tris_verts.data_ptr<scalar_t>(),
                    cand_a_idx.data_ptr<long>(), cand_t_idx.data_ptr<long>(),
                    K,(scalar_t)eps, hit_mask.data_ptr<bool>(),
                    poly_counts.data_ptr<int>(), centroids.data_ptr<scalar_t>(), areas.data_ptr<scalar_t>(),
                    out_a_idx.data_ptr<long>(), out_t_idx.data_ptr<long>());
        });
    } else { // mode == 2
        poly_counts = torch::zeros({K}, opts_i);
        poly_verts  = torch::zeros({K, max_vert, 3}, opts_f);
        centroids   = torch::zeros({K, 3}, opts_f);
        areas       = torch::zeros({K},     opts_f);
        AT_DISPATCH_FLOATING_TYPES(aabbs_min.scalar_type(), "sat_clip_kernel", [&]{
            (max_vert==8 ? sat_clip_kernel<scalar_t,8> : sat_clip_kernel<scalar_t,7>)
                <<<blocks,threads>>>(
                    aabbs_min.data_ptr<scalar_t>(), aabbs_max.data_ptr<scalar_t>(),
                    tris_verts.data_ptr<scalar_t>(),
                    cand_a_idx.data_ptr<long>(), cand_t_idx.data_ptr<long>(),
                    K,(scalar_t)eps, hit_mask.data_ptr<bool>(),
                    poly_counts.data_ptr<int>(),
                    poly_verts.data_ptr<scalar_t>(),
                    centroids.data_ptr<scalar_t>(),
                    areas.data_ptr<scalar_t>(),
                    out_a_idx.data_ptr<long>(), out_t_idx.data_ptr<long>());
        });
    }
    
    return { hit_mask, out_a_idx, out_t_idx, poly_counts, poly_verts, centroids, areas };
}

// --------------------------------------
// Voxelize (原有实现)
// --------------------------------------
template<typename T, bool USE_SAT>
__global__ void k_voxelize_mark(const T* __restrict__ aabb_min, const T* __restrict__ aabb_max, const T* __restrict__ tri_min, const T* __restrict__ tri_max, const T* __restrict__ tris_verts, int64_t Na, int64_t Nt, int64_t a_offset, int64_t t_offset, unsigned char* __restrict__ active_mask, T eps) {
    int ai=blockIdx.x*blockDim.x+threadIdx.x; int ti=blockIdx.y*blockDim.y+threadIdx.y;
    if (ai>=Na || ti>=Nt) return;
    const T* amin=aabb_min+ai*3; const T* amax=aabb_max+ai*3;
    const T* tmin=tri_min +ti*3; const T* tmax=tri_max +ti*3;
    bool ov=(amin[0] <= tmax[0] + eps && amax[0] + eps >= tmin[0]) &&
        (amin[1] <= tmax[1] + eps && amax[1] + eps >= tmin[1]) &&
        (amin[2] <= tmax[2] + eps && amax[2] + eps >= tmin[2]);
    if (!ov) return;
    if constexpr (USE_SAT){
        T center[3]={(amin[0]+amax[0])*T(0.5),(amin[1]+amax[1])*T(0.5),(amin[2]+amax[2])*T(0.5)};
        T he[3]={(amax[0]-amin[0])*T(0.5),(amax[1]-amin[1])*T(0.5),(amax[2]-amin[2])*T(0.5)};
        const T* tv0=tris_verts+(ti*3+0)*3; const T* tv1=tris_verts+(ti*3+1)*3; const T* tv2=tris_verts+(ti*3+2)*3;
        T tri[3][3]={{tv0[0],tv0[1],tv0[2]},{tv1[0],tv1[1],tv1[2]},{tv2[0],tv2[1],tv2[2]}};
        if (!tri_aabb_sat<T>(center,he,tri,eps)) return;
    }
    active_mask[a_offset + ai] = 1;
}

void voxelize_mark_cuda(
    at::Tensor aabb_min, at::Tensor aabb_max,
    at::Tensor tri_min,  at::Tensor tri_max,
    at::Tensor tris_verts,
    int64_t a_offset, int64_t t_offset,
    at::Tensor active_mask,
    bool use_sat, double eps)
{
    const auto Na=aabb_min.size(0), Nt=tri_min.size(0);
    dim3 blk(32,32), grd((Na+blk.x-1)/blk.x, (Nt+blk.y-1)/blk.y);
    AT_DISPATCH_FLOATING_TYPES(aabb_min.scalar_type(), "k_voxelize_mark", [&]{
        if (use_sat){
            k_voxelize_mark<scalar_t,true><<<grd,blk>>>(
                aabb_min.data_ptr<scalar_t>(), aabb_max.data_ptr<scalar_t>(),
                tri_min.data_ptr<scalar_t>(),  tri_max.data_ptr<scalar_t>(),
                tris_verts.data_ptr<scalar_t>(),
                Na,Nt,a_offset,t_offset, active_mask.data_ptr<unsigned char>(), (scalar_t)eps);
        }else{
            k_voxelize_mark<scalar_t,false><<<grd,blk>>>(
                aabb_min.data_ptr<scalar_t>(), aabb_max.data_ptr<scalar_t>(),
                tri_min.data_ptr<scalar_t>(),  tri_max.data_ptr<scalar_t>(),
                tris_verts.data_ptr<scalar_t>(),
                Na,Nt,a_offset,t_offset, active_mask.data_ptr<unsigned char>(), (scalar_t)eps);
        }
    });
}