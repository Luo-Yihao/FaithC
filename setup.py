from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='faithcontour._C',
        sources=[
            'src/faithcontour/_C/bindings.cpp',
            'src/faithcontour/_C/kernels.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '-gencode=arch=compute_70,code=sm_70', # NVIDIA Volta (V100)
                '-gencode=arch=compute_75,code=sm_75', # NVIDIA Turing (RTX 20xx, T4)
                '-gencode=arch=compute_80,code=sm_80', # NVIDIA Ampere (A100)
                '-gencode=arch=compute_86,code=sm_86', # NVIDIA Ampere (RTX 30xx)
                '-gencode=arch=compute_90,code=sm_90', # NVIDIA Hopper (H100) - for future-proofing
            ]
        }
    )
]

setup(
    name='faithcontour',
    version='0.1.0',
    author='Yihao Luo',
    description='A CUDA-accelerated library for Faithful Contouring, with FCT extraction and remeshing.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'trimesh',
        'numpy',
        'scipy',
    ],
)