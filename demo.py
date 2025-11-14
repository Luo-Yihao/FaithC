import os
import torch
import trimesh
import time
import argparse
import sys
import math
import numpy as np
import open3d as o3d

import faithcontour as fc



import contextlib
@contextlib.contextmanager
def SuppressPrint(turn_stdout=True):
    """
    A context manager to temporarily suppress print statements.

    Usage:
    with SuppressPrint():
        noisy_function()
    """
    if not turn_stdout:
        yield
        return
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout

def main():
    """
    Main function to run the FCT encoding and decoding process.
    """
    parser = argparse.ArgumentParser(
        description="Run the FCT hierarchical voxelization and remeshing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 格式化帮助信息
    )
    parser.add_argument(
        "-p", "--mesh_path",
        type=str, 
        default="",
        help="Path to the input mesh file (e.g., /path/to/model.obj)."
    )
    parser.add_argument(
        "-r", "--res",
        type=int, 
        default=512, 
        help="Final grid resolution. Must be a power of two."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="output/reconstructed_mesh.glb",
        help="Path for the output reconstructed mesh file."
    )
    parser.add_argument(
        "--rescalar",
        type=float,
        default=0.95,
        help="Rescaling factor for mesh normalization."
    )
    parser.add_argument(
        "--if_rotate_to_diag",
        type=bool,
        default=False,
        help="Whether to rotate the mesh to align with the diagonal."
    )
    parser.add_argument(
        "--if_simplify",
        type=bool,
        default=False,
        help="Whether to simplify the output mesh using quadric edge collapse decimation."
    )
    parser.add_argument(
        "--clamp_anchors",
        type=bool,
        default=True,
        help="Whether to clamp anchor points to prevent ill-posed solutions. "
             "True ensures stability by constraining anchors within voxel bounds, "
             "False allows unconstrained optimization for potentially higher accuracy."
    )
    args = parser.parse_args()

    # Ensure resolution is a power of two
    if (args.res & (args.res - 1)) != 0 or args.res == 0:
        print(f"Error: Resolution --res must be a power of two, but got {args.res}.")
        sys.exit(1)

    # Set max and min levels based on resolution
    max_level = int(math.log2(args.res))
    min_level = 4 # Minimum level for the octree 
    if max_level <= min_level:
        min_level = max(1, max_level - 1)
    

    print("--- Configuration ---")
    print(f"  Resolution: {args.res} (Max Level: {max_level})")

    if args.if_rotate_to_diag:
        assert 1==0, "Rotation to diagonal is currently disabled."
        print("🦎🦎Rotating mesh to align with the diagonal direction...")

    # --- 3. load and normalize the input mesh ---
    if args.mesh_path == "":
        print("No input mesh path provided. Using a default spherical polyhedron.")
        mesh_tem = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    else:
        try:
            mesh_tem = trimesh.load(args.mesh_path, force='mesh')
            print(f"Loading mesh from: {args.mesh_path}")
        except Exception as e:
            print(f"Error loading mesh: {e}")
            sys.exit(1)

    
    # Generate a random rotation matrix
    # rotation_matrix = trimesh.transformations.random_rotation_matrix()
    # Apply the transformation
    # mesh_tem.apply_transform(rotation_matrix)
    if args.if_rotate_to_diag:
        mesh_tem, _, _ = fc.normalize_mesh_max_fill(mesh_tem, margin=(1-args.rescalar)/2)
    else:
        mesh_tem = fc.normalize_mesh(mesh_tem, rescalar=args.rescalar)
    # mesh_tem = o3d.geometry.TriangleMesh(
    #     vertices=o3d.utility.Vector3dVector(mesh_tem.vertices),
    #     triangles=o3d.utility.Vector3iVector(mesh_tem.faces)
    # )
    # mesh_tem.merge_close_vertices(1/args.res*2)
    # mesh_tem.compute_triangle_normals()

    # Convert mesh data to torch tensors
    V = torch.tensor(np.asarray(mesh_tem.vertices), dtype=torch.float32)
    F = torch.tensor(np.asarray(mesh_tem.faces), dtype=torch.int64)
    N_F = torch.tensor(np.asarray(mesh_tem.face_normals), dtype=torch.float32)
    
    # --- 4. Call FCT Encoder ---
    solver_weights = {'lambda_n': 1.0, 'lambda_d': 1e-3, 'weight_power': 0.5}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    time0 = time.time()
    with SuppressPrint(turn_stdout=False):
        FCT_dict = fc.FCT_encoder(
            vertices=V,
            faces=F,
            face_normals=N_F,
            max_level=max_level,
            min_level=min_level,
            solver_weights=solver_weights,
            device=device,
            output_mode='dict',
            clamp_anchors=args.clamp_anchors  # Control anchor clamping via command line argument
        )
    print(f"FCT encoder time: {time.time()-time0:.3f}s")
    for k, v in FCT_dict.items():
        print(k, ':', v.shape)

    # --- 5. Call FCT Decoder ---
    time0 = time.time()
    recon_points, recon_faces = fc.FCT_decoder(
        FCT_dict,
        resolution=args.res
    )
    
    
    print(f"FCT decoder time: {time.time()-time0:.3f}s")
    print(f"Generated {len(recon_faces)} faces from primal-dual pairs.")


    # --- 6. Export the final mesh ---
    if args.if_simplify:
        import pyfqmr
        print("Simplifying mesh...")
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(recon_points.cpu().numpy(), recon_faces.cpu().numpy())
        mesh_simplifier.simplify_mesh_lossless(verbose=False, epsilon=1e-8)
        vertices, faces, normals = mesh_simplifier.getMesh()
        mesh_out = trimesh.Trimesh(vertices, faces, process=False)
        print(f"Simplified mesh to {len(faces)} faces using quadric edge collapse decimation.")

    else:
        mesh_out = trimesh.Trimesh(recon_points.cpu().numpy(), recon_faces.cpu().numpy(), process=False)
    
    try:
        mesh_out.export(args.output)
        print(f"\n✅ Successfully exported final mesh to '{args.output}'")
    except Exception as e:
        print(f"\nError exporting mesh to '{args.output}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()