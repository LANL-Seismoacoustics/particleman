# vim: set expandtab ts=4 sw=4:

# This file is part of the NeuroImaging Analysis Framework
# For copyright and redistribution details, please see the COPYING file

try:
    import setuptool
except:
    pass

import os
import os.path
import sys
import commands
from glob import glob
from numpy.distutils.core import setup, Extension
import numpy as np

#from distutils.core import setup, Extension

#
# Find the numpy headers
#
try:
    import numpy
except ImportError:
    print "We require the numpy python module to build"
    sys.exit(1)

from numpy.distutils.system_info import get_info, system_info

incdirs = system_info().get_include_dirs()
libdirs = system_info().get_lib_dirs()

# Extension modules
#incdirs += [os.path.join(numpy.__path__[0], 'core/include')]
incdirs += [np.get_include() + '/numpy']

ext_modules = []

fftw = get_info('fftw3')

if not fftw:
    fftw = get_info('fftw')

if not fftw:
    print "We require either fftw3 or fftw to be present in order to build"
    sys.exit(1)

#define_macros = fftw['define_macros'],
ext_modules += [Extension('stockwell.st',
                          include_dirs = incdirs + fftw['include_dirs'],
                          libraries = fftw['libraries'],
                          library_dirs = fftw['library_dirs'],
                          sources = ['stockwell/src/stmodule.c', 'stockwell/src/st.c'])]
ext_modules += [Extension('stockwell.sine',
                          include_dirs = incdirs,
                          sources = ['stockwell/src/sinemodule.c'])]


setup(name = 'stockwell',
      version = '0.0.1',
      description = 'Stockwell module from the NeuroImaging Analysis Framework',
      author = 'YNiC Staff',
      author_email = 'ynic-devel@ynic.york.ac.uk',
      long_description = '''
Stockwell transform module from the YNiC Analysis and Visualisation Tools
''',
       packages = ['stockwell'],
       py_modules = ['stockwell.smt', 'stockwell.util'],
       ext_modules = ext_modules,
)

