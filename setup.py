from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="chPool",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "chPool",
            ["pytorch/chPool_ops.cpp", "kernel/chPool_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
