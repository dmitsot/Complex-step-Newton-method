"""
setup.py — handles the pybind11 C++ extension with optional OpenMP.

OpenMP is enabled when available:
  - macOS   : requires `brew install libomp`; silently skipped if absent
  - Linux   : enabled via -fopenmp
  - Windows : enabled via /openmp

The C++ code already guards every OpenMP call with #ifdef _OPENMP, so the
package builds and runs correctly without OpenMP (single-threaded).
"""

import os
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_compile_args = ["-O2"]
extra_link_args: list[str] = []

if sys.platform == "darwin":
    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if os.path.isfile(os.path.join(prefix, "lib", "libomp.dylib")):
            extra_compile_args += [
                "-Xpreprocessor", "-fopenmp",
                f"-I{prefix}/include",
            ]
            extra_link_args += [f"-L{prefix}/lib", "-lomp"]
    except Exception:
        pass  # libomp not found — OpenMP disabled, pragmas are silently ignored
elif sys.platform.startswith("linux"):
    extra_compile_args += ["-fopenmp"]
    extra_link_args    += ["-fopenmp"]
elif sys.platform == "win32":
    extra_compile_args += ["/openmp"]

ext_modules = [
    Pybind11Extension(
        "csnewton._csnewton",
        sources=["src/csnewton/_csnewton.cpp"],
        include_dirs=["src/csnewton"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
