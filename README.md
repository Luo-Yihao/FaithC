
<div align="center">

# Faithful Contouring: Near-Lossless 3D Voxel Representation Free from Iso-surface

### Enough with SDF + Marching Cubes? &nbsp; Time to bring geometry back — faithfully.

<p>
  <a href="https://arxiv.org/abs/2511.04029"><img src="https://img.shields.io/badge/arXiv-2511.04029-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Luo-Yihao/FaithC"><img src="https://img.shields.io/badge/CVPR%202026-Oral-4b44ce.svg" alt="CVPR 2026 Oral"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

[Yihao Luo](https://luo-yihao.github.io/)<sup>1*</sup>,
[Xianglong He](https://github.com/XianglongHe)<sup>2*</sup>,
[Chuanyu Pan](https://pptrick.github.io/)<sup>3</sup>,
[Yiwen Chen](https://github.com/buaacyw)<sup>3,4</sup>,
Jiaqi Wu<sup>5</sup>,
[Yangguang Li](https://yg256li.github.io/)<sup>6</sup>,
[Wanli Ouyang](https://wlouyang.github.io/)<sup>6</sup>,
[Yuanming Hu](https://github.com/yuanming-hu)<sup>3</sup>,
[Guang Yang](https://www.yanglab.fyi/)<sup>1</sup>,
[ChoonHwai Yap](https://yaplab.github.io/)<sup>1</sup>

<sup>1</sup>Imperial College London &nbsp;
<sup>2</sup>Tsinghua University &nbsp;
<sup>3</sup>Meshy &nbsp;
<sup>4</sup>Nanyang Technological University &nbsp;
<sup>5</sup>University of Melbourne &nbsp;
<sup>6</sup>The Chinese University of Hong Kong


![Teaser](imgs/Cover_FCT.png)

</div>

## News

- **[2026-03]** 🎉 Accepted as **Oral** at **CVPR 2026**!
- **[2026-01]** 🤝 Concurrent work [TRELLIS 2](https://github.com/microsoft/TRELLIS.2) released with [O-Voxel](https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel) representation — great to see the community moving beyond iso-surfaces.
- **[2025-12]** 🚀 Code fully open-sourced! v1.5 released — pure Python + Atom3d, no C++ compilation required.
- **[2025-11]** 📄 arXiv preprint and wheel package released.


## Overview

Conventional voxel-based mesh representations rely on distance fields (SDF/UDF) and iso-surface extraction through Marching Cubes. These pipelines require watertight preprocessing and global sign computation, often introducing artifacts like surface thickening, jagged iso-surfaces, and loss of internal structures.

**Faithful Contouring** avoids these issues by directly operating on the raw mesh. It identifies all surface-intersecting voxels and solves for a compact set of local anchor features — **Faithful Contour Tokens (FCTs)** — that enable near-lossless reconstruction.

- **High fidelity** — sharp edges and internal structures preserved, even for open or non-manifold meshes
- **Scalable** — efficient GPU kernels enable resolutions up to 2048+
- **Compact** — 18 dimensions per voxel token
- **Flexible** — token format supports filtering, texturing, manipulation, and assembly

![Compare](imgs/WUKONGCOMPARE.png)


## Installation

### Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- PyTorch 2.5+

### Quick Start with Pixi (Recommended)

[Pixi](https://pixi.sh) handles all dependencies automatically — one command to install, one to run:

```bash
git clone https://github.com/Luo-Yihao/FaithC.git
cd FaithC
pixi run demo
```

That's it. The first run installs everything (Python, PyTorch, torch-scatter, Atom3d, etc.) and runs the demo. Subsequent runs take ~5 seconds.

<details>
<summary><b>Install Pixi</b></summary>

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

</details>

### Manual Setup

```bash
# Install PyTorch (match your CUDA version, example for CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install torch_scatter
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Install Atom3d (geometry backend)
pip install git+https://github.com/Luo-Yihao/Atom3d.git --no-build-isolation

# Install FaithContour
git clone https://github.com/Luo-Yihao/FaithC.git
cd FaithC
pip install -e . --no-build-isolation

# Other dependencies
pip install trimesh scipy einops
```


## Quick Start

```bash
# Default icosphere at resolution 128
python demo.py

# Custom mesh at resolution 512
python demo.py -p assets/examples/pirateship.glb -r 512 -o output/pirateship.glb
```

<details>
<summary><b>All arguments</b></summary>

| Argument | Default | Description |
|---|---|---|
| `-p, --mesh_path` | `""` (icosphere) | Path to input mesh |
| `-r, --res` | `128` | Grid resolution (power of 2) |
| `-o, --output` | `output/reconstructed_mesh.glb` | Output path |
| `--margin` | `0.05` | Grid boundary margin |
| `--tri_mode` | `auto` | Triangulation mode: `auto`, `length`, `angle`, `normal_abs`, `simple_02`, `simple_13` |
| `--clamp_anchors` | `True` | Clamp anchors to voxel bounds |
| `--compute_flux` | `True` | Compute edge flux signs |

</details>


## API Usage

```python
import torch
import trimesh
from faithcontour import FCTEncoder, FCTDecoder
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer

# Load mesh
mesh = trimesh.load("model.obj", force='mesh')
V = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
F = torch.tensor(mesh.faces, dtype=torch.long, device='cuda')

# Build spatial structures
bvh = MeshBVH(V, F)
bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]], device='cuda')
octree = OctreeIndexer(max_level=9, bounds=bounds, device='cuda')  # 512^3

# Encode
encoder = FCTEncoder(bvh, octree, device='cuda')
fct = encoder.encode(min_level=4, compute_flux=True, clamp_anchors=True)
# fct.anchor:             [K, 3]  — surface anchor points
# fct.normal:             [K, 3]  — surface normals
# fct.edge_flux_sign:     [K, 12] — edge crossing signs {-1, 0, +1}
# fct.active_voxel_indices: [K]   — linear voxel indices

# Decode
decoder = FCTDecoder(resolution=512, bounds=bounds, device='cuda')
result = decoder.decode_from_result(fct)

# Export
trimesh.Trimesh(
    result.vertices.cpu().numpy(),
    result.faces.cpu().numpy()
).export("output.glb")
```


## FCT Token Format

Each active voxel is encoded as an 18-dimensional token:

| Field | Dims | Type | Description |
|---|---|---|---|
| `anchor` | 3 | float32 | Surface representative point |
| `normal` | 3 | float32 | Surface normal direction |
| `edge_flux_sign` | 12 | int8 | Edge crossing signs {-1, 0, +1} |


## How It Works

**Encoder** — mesh to tokens:
1. Hierarchical octree traversal with BVH-accelerated AABB intersection
2. SAT polygon clipping at the finest level for precise centroids and areas
3. QEF (Quadric Error Function) solve for optimal anchor points and normals
4. Segment-triangle intersection for edge flux sign computation

**Decoder** — tokens to mesh:
1. Identify edges with non-zero flux (surface crossings)
2. Form quads from 4 voxels incident to each active edge
3. Adaptive triangulation based on normal consistency


## Performance

Benchmarked on NVIDIA H100:

| Resolution | Active Voxels | Encode | Decode | Total |
|---|---|---|---|---|
| 128 | 71K | 0.27s | 0.02s | 0.29s |
| 256 | 287K | 0.45s | 0.06s | 0.51s |
| 512 | 1.1M | 0.52s | 0.17s | 0.70s |
| 1024 | 4.6M | 0.82s | 0.61s | 1.42s |
| 2048 | 18.4M | 2.16s | 2.51s | 4.68s |


## Roadmap

- [x] Wheel package for Linux (v0.1)
- [x] Pure Python + Atom3d implementation (v1.5)
- [ ] FCT-based VAE release
- [ ] Diffusion model release


## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{luo2026faithfulcontouring,
    title     = {Faithful Contouring: Near-Lossless 3D Voxel Representation Free from Iso-surface},
    author    = {Luo, Yihao and He, Xianglong and Pan, Chuanyu and Chen, Yiwen and Wu, Jiaqi and Li, Yangguang and Ouyang, Wanli and Hu, Yuanming and Yang, Guang and Yap, ChoonHwai},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```


## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).


## Contact

Yihao Luo — [y.luo23@imperial.ac.uk](mailto:y.luo23@imperial.ac.uk)

Project: [https://github.com/Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC)
