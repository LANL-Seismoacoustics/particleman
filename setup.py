#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Particle Man: particle motion analysis of seismic surface waves

"""
import os
import sys
import pathlib

from numpy.distutils.system_info import get_info, system_info
from setuptools import setup, Extension, find_packages

here = pathlib.Path(__file__).parent.resolve()
readme = (here / 'README.md').read_text(encoding='utf-8')

# Find the fftw3 headers
info = system_info()
incdirs = info.get_include_dirs()
libdirs = info.get_lib_dirs()

fftw = get_info('fftw3') or get_info('fftw')

if not fftw:
    print("We require either fftw3 or fftw to be present in order to build", file=sys.stderr)
    sys.exit(1)

#define_macros = fftw['define_macros'],
ext_modules = [Extension('particleman.libst',
                         include_dirs=incdirs + fftw['include_dirs'],
                         libraries=fftw['libraries'],
                         library_dirs=fftw['library_dirs'],
                         sources=['src/particleman/st.c'])]


setup(name='particleman',
      version='0.3.0',
      description='Particle motion analysis for seismic surface waves',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://github.com/lanl-seismoacoustics/particleman',
      author='Jonathan MacCarthy',
      author_email='jkmacc@lanl.gov',
      install_requires=['numpy', 'matplotlib'],
      python_requires='>=3.4',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      ext_modules=ext_modules,
     )
