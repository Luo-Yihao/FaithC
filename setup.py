# Builds the faithcontour._C extension. On a CUDA PyTorch this compiles the
# original CUDA sources; on a ROCm PyTorch BuildExtension hipifies them so the
# same sources build for AMD GPUs.

import os
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_C_DIR = os.path.join("src", "faithcontour", "_C")

sources = [
    os.path.join(_C_DIR, "bindings.cpp"),
    os.path.join(_C_DIR, "kernels.cu"),
]
extra_link_args = []
if sys.platform == "win32":
    # c10.dll does not export the c10::ValueError(SourceLocation, string)
    # constructor that is inherited via "using Error::Error".  Headers included
    # through <torch/extension.h> (e.g. ATen/TensorIndexing.h) trigger
    # TORCH_CHECK_VALUE which generates a __declspec(dllimport) reference to
    # that constructor, causing LNK2001.  Alias it to Error(SourceLocation,
    # string) which IS exported from c10.dll.  ValueError IS-A Error with no
    # extra data members; the constructors are semantically identical.
    # Alias the missing dllimport thunk for ValueError(SourceLocation,string)
    # to Error(SourceLocation,string) which IS in c10.dll.
    _val_imp = (
        "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    _err_imp = (
        "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")

setup(
    name="faithcontour",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[
        CUDAExtension(
            name="faithcontour._C",
            sources=sources,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
