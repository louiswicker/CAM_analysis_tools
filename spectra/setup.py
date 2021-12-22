#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

#include_dirs=[np.get_include()])
#   extra_compile_args=['-I/Users/jeremy.gibbs/opt/anaconda3/lib/python3.8/site-packages/numpy/core/include'],)]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("spectra", ["spectra.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=['-I/np.get_include()'],)]
)
