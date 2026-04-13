import torch
from typing import Dict, Union, Tuple
from .utils.grid import *
from .ops import (
    find_intersections_select,
    solve_points_by_group,
    segment_triangle_intersections_fast,
)
from .utils.mesh import *
import time
import einops as eins
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_min, scatter_softmax
import cubvh

@torch.no_grad()
def FCT_encoder(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_level: int,
    face_normals: torch.Tensor = None,
    vertex_normals: torch.Tensor = None,
    min_level: int = 4,
    solver_weights: Dict[str, float] = {
        'lambda_n': 1.0, 'lambda_d': 0.1, 'weight_power': 1.0, 'eps': 1e-9
    },
    device: str = 'cuda',
    use_primal_verts: bool = False,
    output_mode: str = 'dict', # 'dict' | 'fct' | 'mesh'
    clamp_anchors: bool = True
) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Performs hierarchical voxelization and can directly output Faithful Contour Tokens (FCT).
    """
    print("--- Starting Hierarchical Voxelization ---")
    
    start_time = time.time()
    indexer = OctreeIndexer(max_level=max_level, device='cpu')
    
    triangles = vertices[faces].to(device, torch.float32)
    
    BVH = cubvh.cuBVH(vertices.cpu().numpy(), faces.cpu().numpy())

    if face_normals is None:
        face_normals = torch.nn.functional.normalize(
            torch.cross(
                triangles[:,1,:] - triangles[:,0,:],
                triangles[:,2,:] - triangles[:,0,:]
            ),
            dim=1
        )
        face_normals = face_normals/face_normals.norm(dim=1, keepdim=True).clamp_min(1e-9)
    
    if vertex_normals is None:
        vertex_normals = torch.zeros_like(vertices, dtype=torch.float32)
        vertex_normals.index_add_(0, faces[:,0], face_normals)
        vertex_normals.index_add_(0, faces[:,1], face_normals)
        vertex_normals.index_add_(0, faces[:,2], face_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
        vertex_normals = vertex_normals/vertex_normals.norm(dim=1, keepdim=True).clamp_min(1e-9)


    
    num_cells_min_level = (1 << min_level)**3
    candidate_mortons = indexer.linear_to_morton(
        torch.arange(num_cells_min_level, device=device, dtype=indexer.index_dtype),
        level=min_level
    )


    for level in range(min_level, max_level + 1):
        level_start_time = time.time()
        print(f"\n--- Processing Level {level}/{max_level} ---")

        if candidate_mortons.numel() == 0:
            print("No candidates left. Stopping early.")
            break
            
        print(f"Number of candidate voxels for this level: {len(candidate_mortons)}")

        candidate_linear_indices_level = indexer.morton_to_linear(candidate_mortons, level)
        aabbs_min, aabbs_max = indexer.primal_cube_aabbs_minmax(
            cube_idx=candidate_linear_indices_level, level=level
        )
        
        if level < max_level:
            hit_indices_relative = find_intersections_select(
                aabbs_min, aabbs_max, triangles, return_mode="aabb", keep_on=device
            )
            active_mortons = candidate_mortons[hit_indices_relative.cpu()]
            print(f"Found {len(active_mortons)} active voxels at level {level}.")
            
            if active_mortons.numel() > 0:
                candidate_mortons = torch.unique(indexer.children_morton(active_mortons.cpu()).flatten())
            else:
                candidate_mortons = torch.tensor([], dtype=torch.int64, device=device)
        else:
            print("Final level: switching to 'centroid' mode for precise feature calculation.")
            ## ================Process primal cubes =================
            if use_primal_verts:

                a_idx_rel, t_idx, centroids, areas = find_intersections_select(
                    aabbs_min, aabbs_max, triangles, return_mode="centroid", keep_on=device
                )
                
                if a_idx_rel.numel() == 0:
                    print("No active voxels found at the final level after centroid check. Exiting.")
                    break


                a_idx_global = candidate_linear_indices_level[a_idx_rel.cpu()]
                
                normals_t_idx = face_normals[t_idx.cpu(), :].to(device)

                active_voxels_indices, p_vox, n_vox = solve_points_by_group(
                    a_idx_global.cuda(), centroids.cuda(), normals_t_idx.cuda(), areas.cuda(), **solver_weights
                )
                print(f"Solved for {len(active_voxels_indices)} unique active primal voxels.")
                del aabbs_min, aabbs_max, a_idx_rel, t_idx, centroids, areas
            else:
                active_voxels_indices = find_intersections_select(
                    aabbs_min, aabbs_max, triangles, return_mode="aabb", keep_on=device
                )
                active_voxels_indices = candidate_linear_indices_level[active_voxels_indices.cpu()]
                p_vox = indexer.primal_cube_aabbs_centers(active_voxels_indices).cpu()
                n_vox = torch.zeros_like(p_vox, device=device)
                print(f"Using voxel centers for {len(active_voxels_indices)} active primal voxels.")
                del aabbs_min, aabbs_max
            
            ## ==========Process dual cubes =================
            primal_dual_map = indexer.primal_cubes_to_dual_cubes_indices(active_voxels_indices.cpu()).contiguous()

            candidate_dual_unique, candidate_dual_inv = torch.unique(primal_dual_map.flatten().cuda(),
                                                                                  return_inverse=True)
            candidate_dual_unique = candidate_dual_unique.cpu()
            print(f"Found {len(candidate_dual_unique)} unique dual neighbors to process.")

            dual_aabbs_min, dual_aabbs_max = indexer.dual_cube_aabbs_minmax(candidate_dual_unique)
            dual_aabbs_min, dual_aabbs_max = dual_aabbs_min.contiguous(), dual_aabbs_max.contiguous()


            d_a_idx_candy, d_t_idx, d_centroids, d_areas = find_intersections_select(
                dual_aabbs_min, dual_aabbs_max, triangles, return_mode="centroid", keep_on=device
            )

            del dual_aabbs_min, dual_aabbs_max
            torch.cuda.empty_cache()

            normals_d_t_idx = face_normals[d_t_idx.cpu(), :].to(device)

            active_dual_local_unique, p_vox_dual, n_vox_dual = solve_points_by_group(
                d_a_idx_candy.cuda(), 
                d_centroids.cuda(),
                normals_d_t_idx.cuda(),
                d_areas.cuda(),
                **solver_weights,
            )

            p_vox_dual = torch.nan_to_num(p_vox_dual, nan=0.0, posinf=0.0, neginf=0.0)
            n_vox_dual = torch.nan_to_num(n_vox_dual, nan=0.0, posinf=0.0, neginf=0.0)
            
            
            with torch.no_grad():
                if clamp_anchors:
                    solved_dual_aabbs_min, solved_dual_aabbs_max = indexer.dual_cube_aabbs_minmax(candidate_dual_unique[active_dual_local_unique.cpu()])
                    p_vox_dual_clamped = torch.clamp(p_vox_dual, solved_dual_aabbs_min.to(p_vox_dual.device),
                                                    solved_dual_aabbs_max.to(p_vox_dual.device))

                    _, udf_faces_idx, udf_uvw = BVH.unsigned_distance(p_vox_dual_clamped.view(-1, 3), return_uvw=True)
                    udf_uvw = torch.nan_to_num(udf_uvw, nan=0.0, posinf=1.0, neginf=0.0)
                    udf_uvw = udf_uvw.clamp(min=1e-9, max=1.0)
                    udf_uvw = udf_uvw / (udf_uvw.sum(dim=-1, keepdim=True))

                    trig_center = triangles[udf_faces_idx].mean(dim=-2, keepdim=True)
                    reloc_pox = trig_center.squeeze(-2) + ((triangles[udf_faces_idx]-trig_center)* udf_uvw.unsqueeze(-1).to(p_vox_dual.device)).sum(dim=-2)
                    
                
                    p_vox_dual = torch.where((p_vox_dual_clamped-p_vox_dual).norm(dim=-1, keepdim=True)<1e-8, 
                                                    p_vox_dual_clamped, reloc_pox)
                    
                    dual_voxel_center = indexer.dual_cube_aabbs_centers(candidate_dual_unique[active_dual_local_unique.cpu()]).to(p_vox_dual.device)


                    p_vox_dual_rel = p_vox_dual - dual_voxel_center

                    p_vox_dual_rel = torch.clamp(p_vox_dual_rel, -1/(2**max_level), +1/(2**max_level))

                    p_vox_dual  = p_vox_dual_rel + dual_voxel_center
                    del solved_dual_aabbs_min, solved_dual_aabbs_max, p_vox_dual_clamped, udf_faces_idx, udf_uvw, reloc_pox, dual_voxel_center, p_vox_dual_rel
                    torch.cuda.empty_cache()
                
            print('organizing FCT data...')

            candy_local_mask = torch.zeros(candidate_dual_unique.shape[0], dtype=torch.bool, device=device)
            candy_local_mask[active_dual_local_unique] = True

            candy_local_features = torch.zeros((candidate_dual_unique.shape[0], 6), dtype=p_vox_dual.dtype, device=device)
            candy_local_features[active_dual_local_unique, :3] = p_vox_dual
            candy_local_features[active_dual_local_unique, 3:] = n_vox_dual

            candy_local_indices = torch.full((candidate_dual_unique.shape[0],), -1, dtype=indexer.index_dtype, device=device)
            candy_local_indices[active_dual_local_unique] = torch.arange(active_dual_local_unique.shape[0], device=device, dtype=indexer.index_dtype)

            ## Get the active primal-dual mapping
            primal_dual_mask = candy_local_mask[candidate_dual_inv]
            primal_dual_features = candy_local_features[candidate_dual_inv]
            primal_local_indices = candy_local_indices[candidate_dual_inv] 

            primal_dual_mask = eins.rearrange(primal_dual_mask, '(K F) -> K F', K=active_voxels_indices.shape[0], F=8)
            primal_dual_features = eins.rearrange(primal_dual_features, '(K F) C -> K F C', K=active_voxels_indices.shape[0], F=8, C=6)
            primal_local_indices = eins.rearrange(primal_local_indices, '(K F) -> K F', K=active_voxels_indices.shape[0], F=8) # [K, 8]

            ## FCT dict
            FCT_dict = {}
            
            FCT_dict['active_voxels_indices'] = active_voxels_indices # [K]
            FCT_dict['primal_anchor'] = p_vox # [K, 3]
            FCT_dict['primal_normal'] = n_vox
            FCT_dict['primal_dual_mask'] = primal_dual_mask # [K, 8]
            FCT_dict['primal_dual_anchor'] = primal_dual_features[..., :3] # [K, 8, 3]
            FCT_dict['primal_dual_normal'] = primal_dual_features[..., 3:] # [K, 8, 3]

            print(f"Solved for {len(p_vox_dual)} unique active dual voxels.")

            ## ====================== Process Semiaxis =======================
            semi_axis_dualverts = indexer.get_semiaxis_dualverts_pairs_from_primal(active_voxels_indices.cpu()) # [K, 6, 2]
            semi_axis_dualverts = eins.rearrange(semi_axis_dualverts, 'b l v -> (b l) v')
            semi_axis_dualverts_sort, semi_axis_dualverts_direction = semi_axis_dualverts.sort(dim=-1)
            # Direction of the unique semi-axis
            semi_axis_dualverts_direction = torch.where(semi_axis_dualverts_direction[:,0]<semi_axis_dualverts_direction[:,1], 1, -1)
            # Inverse map to the unique semi-axis
            semi_axis_candidates, semi_axis_candidates_inverse = torch.unique(semi_axis_dualverts_sort.cuda(),
                                                                              dim=0, return_inverse=True)
            print('starting semiaxis processing...')

            semi_axis_candidates_coords = indexer.dual_vertices_ext_coords_from_indices(semi_axis_candidates.cpu()).contiguous()

            semi_axis_candidates_mid = 0.5 * (semi_axis_candidates_coords[:,0,:] + semi_axis_candidates_coords[:,1,:]) # [S, 3]

            # semi_axis_valid, semi_axis_tri_indices, semi_axis_tri_dot = segment_triangle_intersections_fast(
            #                 semi_axis_candidates_coords.to(device).contiguous(),
            #                 triangles.contiguous(), eps=0.0)

            with torch.no_grad():
                semi_axis_candidates_mid_udf, _, _= BVH.unsigned_distance(semi_axis_candidates_mid, return_uvw=False)

            semi_axis_candidates_sure = torch.where(semi_axis_candidates_mid_udf<=(1/(1<<level)))[0]
            semi_axis_valid, semi_axis_tri_indices, semi_axis_tri_dot = segment_triangle_intersections_fast(
                            semi_axis_candidates_coords.to(device)[semi_axis_candidates_sure].contiguous(),
                            triangles.contiguous(), eps=0.0)
            # Filter out invalid semi-axes
            semi_axis_valid = semi_axis_candidates_sure[semi_axis_valid.cpu()]



            semi_axis_valid_mask = semi_axis_valid & (semi_axis_tri_dot.abs() > 0.0)
            semi_axis_valid_mask_map = semi_axis_valid_mask.nonzero(as_tuple=True)[0]

            # Search the minimum absolute dot product for each valid semi-axis
            semi_axis_valid_mask_map = scatter_min(semi_axis_tri_dot.abs(), semi_axis_valid, dim_size=semi_axis_candidates.shape[0])[1]

            print(f"Total candidate semi-axes to check: {(semi_axis_valid_mask_map < semi_axis_candidates.shape[0]).sum().item()} / {semi_axis_candidates.shape[0]}")


            semi_axis_candidates_dot  = torch.cat([semi_axis_tri_dot, torch.Tensor([0]).to(semi_axis_tri_dot.device)], dim=0)[semi_axis_valid_mask_map.cpu()]

            semi_axis_dualverts_dot = semi_axis_candidates_dot.sign()[semi_axis_candidates_inverse]*semi_axis_dualverts_direction.to(semi_axis_candidates_dot.device)

            semi_axis_dualverts_dot = eins.rearrange(semi_axis_dualverts_dot, '(b l) -> b l', l=6)

            FCT_dict['primal_semiaxis_direction'] = semi_axis_dualverts_dot # [K, 6]


        print(f"Final Level {level} processed in {time.time() - level_start_time:.2f}s.")

    total_time = time.time() - start_time
    print(f"\n--- Hierarchical Voxelization Finished in {total_time:.2f}s ---")
    
        
    if output_mode == 'dict':
        FCT_dict = {k: v.cpu() for k, v in FCT_dict.items()}
        return FCT_dict

    elif output_mode == 'mesh': 
        
        semiaxis_mask = semi_axis_dualverts_dot.abs().to(device)

        active_cube_all_valid_quad = primal_local_indices[:, CUBE_FACES] # [K, 6, 4]

        Pv_index, F_index = torch.where((semiaxis_mask > 0.5) & (active_cube_all_valid_quad > -1).all(dim=-1))

        Quad_faces = active_cube_all_valid_quad[Pv_index, F_index]

        Quad_faces = torch.where(semi_axis_dualverts_dot[Pv_index, F_index, None]>0, Quad_faces, Quad_faces.flip(-1))

        Quad_faces_unique = filter_duplicate_faces(Quad_faces)

        V_recon = p_vox_dual.cpu()
        faces_tri = triangulate_quads_by_angle(V_recon, Quad_faces_unique.cpu())

        return V_recon.cpu(), faces_tri.cpu()

    elif output_mode == 'fct':
        faithcontour_token = torch.cat([
            FCT_dict['active_voxels_indices'].unsqueeze(-1).to(torch.float32), # [K, 1]
            FCT_dict['primal_anchor'], # [K, 3]
            FCT_dict['primal_normal'], # [K, 3]
            FCT_dict['primal_dual_mask'].to(torch.float32), # [K, 8]
            eins.rearrange(FCT_dict['primal_dual_anchor'], 'K D C -> K (D C)', C=3), # [K, 24]
            eins.rearrange(FCT_dict['primal_dual_normal'], 'K D C -> K (D C)', C=3), # [K, 24]
            FCT_dict['primal_semiaxis_direction'].to(torch.float32) # [K, 6]
        ], dim=-1)
        return faithcontour_token # [K, 1+3+3+8+24+24+6=69]

    else:
        raise ValueError("Invalid output_mode. Choose 'fct' or 'dict' or 'mesh'.")


def faithcontour_token_to_dict(faithcontour_token: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Converts a FCT  back into a dictionary format.
    """
    if faithcontour_token.ndim != 2 or faithcontour_token.shape[1] != 75:
        raise ValueError("Input tensor must have shape [K, 75].")
    
    FCT_dict = {}
    FCT_dict['active_voxels_indices'] = faithcontour_token[:, 0].round().to(torch.int64)
    FCT_dict['primal_anchor'] = faithcontour_token[:, 1:4]
    FCT_dict['primal_normal'] = faithcontour_token[:, 4:7]
    FCT_dict['primal_dual_mask'] = (faithcontour_token[:, 7:15] > 0.5).to(torch.bool)
    FCT_dict['primal_dual_anchor'] = eins.rearrange(faithcontour_token[:, 15:39], 'K (D C) -> K D C', D=8, C=3)
    FCT_dict['primal_dual_normal'] = eins.rearrange(faithcontour_token[:, 39:63], 'K (D C) -> K D C', D=8, C=3)
    FCT_dict['primal_flux_mask'] = (faithcontour_token[:, 63:75] > 0.5).to(torch.bool)

    return FCT_dict



def FCT_decoder(
    FCT_dict: Dict,
    resolution: int,
    device: str = 'cuda'):

    """
    Extracts dual mesh from saved 
        - active_dual_cubes_indices: [M,] Unique active dual voxel global indices.
        - primal_anchor: [K, 3] Reconstructed primal features (absolute positions).
        - primal_normal: [K, 3] Reconstructed primal normals (absolute directions).
        - primal_dual_mask: [K, 8] Validity mask for the dual neighbors of each primal voxel.
        - primal_dual_anchor: [K, 8, 3] Reconstructed dual features (absolute positions).
        - primal_dual_normal: [K, 8, 3] Reconstructed dual normals (absolute directions).
        - semiaxis_direction: [K, 6] Direction of the semi-axes, {-1, 1}, 0 for invalid.
    returns:
        - V_recon: [M, 3] Reconstructed dual vertices (absolute positions
        - faces_tri: [N, 3] Triangulated faces of the dual mesh.
    """
    indexer = GridExtent(resolution
                         )
    active_voxels_indices = FCT_dict['active_voxels_indices'].to(device)
    p_vox = FCT_dict['primal_anchor'].to(device)
    p_normal = FCT_dict['primal_normal'].to(device)
    p_dual_mask = FCT_dict['primal_dual_mask'].to(device).bool() # K, 8
    p_dual_anchor = FCT_dict['primal_dual_anchor'].to(device) # K, 8, 3
    p_dual_normal = FCT_dict['primal_dual_normal'].to(device) # K, 8, 3
    if 'primal_semiaxis_direction' in FCT_dict:
        semiaxis_direction = FCT_dict['primal_semiaxis_direction'].to(device)
        semiaxis_mask = (semiaxis_direction.abs() > 0.5).to(torch.bool) #K, 6

    else:
        assert 1==0, "Error: The provided Faithful Contour Tokens (FCT) dictionary lacks 'primal_semiaxis_direction'. Please ensure it is included."
        flux_mask = FCT_dict['primal_flux_mask'].to(device) # K, 12
        semiaxis_mask = flux_mask2semiaxis_mask(flux_mask) #K, 6
        semiaxis_mask = (semiaxis_mask >= 4).to(torch.bool)
        semiaxis_direction = semiaxis_mask

    primal_dual_map = indexer.primal_cubes_to_dual_cubes_indices(active_voxels_indices).to(device) # K,8

    dual_unique, dual_unique_inv = torch.unique(primal_dual_map.flatten().cuda(),
                                                return_inverse=True)

    

    ## gather feature

    dual_features = torch.cat([p_dual_anchor, p_dual_normal],dim=-1) # K, 8, F=7
    dual_features = eins.rearrange(dual_features, 'K D F -> (K D) F', D=8)
    dual_masks = eins.rearrange(p_dual_mask, 'K D -> (K D) 1', D=8).float()
    
    dual_features = dual_features * dual_masks # guarantee zero out invalid duals
    dual_features_gather = scatter_sum(dual_features, dual_unique_inv, dim=0, dim_size=dual_unique.shape[0])
    dual_features_gather_mask = scatter_mean(dual_masks, dual_unique_inv, dim=0, dim_size=dual_unique.shape[0])

    dual_features_gather_num = scatter_sum(dual_masks, dual_unique_inv, dim=0, dim_size=dual_unique.shape[0])
    dual_features_gather = dual_features_gather/(dual_features_gather_num+1e-8)

    
    dual_features_gather_mask = (dual_features_gather_mask[..., 0]>0.5) #
    
    primal_local_indices = eins.rearrange(dual_unique_inv, '(K D) -> K D', D=8)
    primal_dual_mask  = eins.rearrange(dual_features_gather_mask[dual_unique_inv], '(K D) -> K D', D=8)

    primal_local_indices = torch.where(primal_dual_mask, primal_local_indices, -1)

    V_recon = dual_features_gather[:,-6:-3] # K'x3

    active_cube_all_valid_quad = primal_local_indices[:, CUBE_FACES]

    Pv_index, F_index = torch.where(semiaxis_mask & (active_cube_all_valid_quad > -1).all(dim=-1))

    Quad_faces = active_cube_all_valid_quad[Pv_index, F_index]

    Quad_faces = torch.where(semiaxis_direction[Pv_index, F_index, None]>0, Quad_faces, Quad_faces.flip(-1))

    Quad_faces_unique_index = filter_duplicate_faces_index(Quad_faces)

    Pv_index_unique = Pv_index[Quad_faces_unique_index].to(Quad_faces.device)

    # Quad_faces_unique = Quad_faces[Quad_faces_unique_index] + p_vox.shape[0] # shift to global index

    # V_recon = torch.cat([p_vox, V_recon], dim=0)

    # faces_tri = torch.cat([Pv_index_unique.view(-1,1), Quad_faces_unique[:, [0, 1]],
    #                         Pv_index_unique.view(-1,1), Quad_faces_unique[:, [1, 2]],
    #                         Pv_index_unique.view(-1,1), Quad_faces_unique[:, [2, 3]],
    #                         Pv_index_unique.view(-1,1), Quad_faces_unique[:, [3, 0]]], dim=-1)

    # faces_tri = eins.rearrange(faces_tri, 'F (T V) -> (F T) V', T=4, V=3)

    Quad_faces_unique = Quad_faces[Quad_faces_unique_index]  # local index

    V_recon = V_recon.cpu()

    faces_tri = triangulate_quads_by_angle(V_recon, Quad_faces_unique.cpu())

    return V_recon.cpu(), faces_tri.cpu()



