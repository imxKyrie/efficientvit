from setuptools import setup, find_packages
from torch.utils import cpp_extension
import os
from pathlib import Path

here = os.path.abspath(os.path.dirname(__file__))
workspace = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))

setup(
    name="litemla_cuda",
    packages=find_packages(where=here),
    package_data={
        "litemla_cuda.ops": ["litemla_ops_cuda*.so"],
    },
    py_modules=["litemla"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="litemla_ops_cuda",
            sources=[
                "litemla.cpp",
                "litemla_kernel.cu",
                "litemla_seqlen1472_stage3.cu",
                "litemla_seqlen1472_stage4.cu",
                "litemla_seqlen1472_stage5.cu",
                "litemla_seqlen1472_stage6.cu",
                "litemla_seqlen368_stage3.cu",
                "litemla_seqlen368_stage4.cu",
                "litemla_seqlen368_stage5.cu",
                "litemla_seqlen368_stage6.cu",
            ],
            include_dirs=[
                Path(workspace) / "third-party" / "cutlass" / "include",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
)
