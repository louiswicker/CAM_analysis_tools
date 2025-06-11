#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as numpy

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules = [Extension("spectra", ["spectra.pyx"],
    extra_compile_args=['-I/home/louis.wicker/miniconda3/envs/gnu12/lib/python3.12/site-packages/numpy/core/include',
    '-I/home/louis.wicker/miniconda3/envs/libs/include/c++/v1'],)]
)
