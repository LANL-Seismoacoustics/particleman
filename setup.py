# vim: set expandtab ts=4 sw=4:

# This file is part of the NeuroImaging Analysis Framework
# For copyright and redistribution details, please see the COPYING file

# https://github.com/synergetics/stockwell_transform

import sys
try:
    import setuptools
except ImportError:
    pass

try:
    import numpy as np
    from numpy.distutils.core import setup, Extension
    from numpy.distutils.system_info import get_info, system_info
except ImportError:
    print("We require the numpy python module to build")
    sys.exit(1)

#
# Find the numpy headers
#
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
      version='0.0.1',
      description='Stockwell module from the NeuroImaging Analysis Framework',
      author='YNiC Staff',
      author_email='ynic-devel@ynic.york.ac.uk',
      long_description='''
Stockwell transform module from the YNiC Analysis and Visualisation Tools
''',
      packages=['stockwell'],
      py_modules=['stockwell.util', 'stockwell.plotting', 'stockwell.st'],
      ext_modules=ext_modules,
     )
