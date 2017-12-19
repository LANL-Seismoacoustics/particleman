#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyStockwell

Stockwell transforms for seismology.

"""
import os
import sys

from numpy.distutils.system_info import get_info, system_info

try:
    import setuptools
except ImportError:
    pass

from distutils.core import setup, Extension

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

try:
    import pypandoc
    readme = pypandoc.convert(readme, 'md', 'rst')
except ImportError:
    pass

# Find the fftw3 headers
info = system_info()
incdirs = info.get_include_dirs()
libdirs = info.get_lib_dirs()

fftw = get_info('fftw3') or get_info('fftw')

if not fftw:
    print("We require either fftw3 or fftw to be present in order to build")
    sys.exit(1)

#define_macros = fftw['define_macros'],
ext_modules = [Extension('stockwell.libst',
                         include_dirs=incdirs + fftw['include_dirs'],
                         libraries=fftw['libraries'],
                         library_dirs=fftw['library_dirs'],
                         sources=['stockwell/src/st.c'])]


setup(name='stockwell',
      version='0.2.0',
      description='Stockwell transforms and applications in seismology',
      long_description=readme,
      url='http://github.com/lanl-seismoacoustics/bulletproof',
      author='Jonathan MacCarthy',
      author_email='jkmacc@lanl.gov',
      install_requires=['numpy'],
      packages=['stockwell'],
      py_modules=['stockwell.util', 'stockwell.plotting', 'stockwell.st'],
      ext_modules=ext_modules,
      data_files=[('tests/data', ['tests/data/BW.RJOB..EHZ.txt'])],
     )
