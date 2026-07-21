#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Particle Man: particle motion analysis of seismic surface waves

"""
import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
readme = (here / 'README.md').read_text(encoding='utf-8')

setup(name='particleman',
      version='0.4.0',
      description='Particle motion analysis for seismic surface waves',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://github.com/lanl-seismoacoustics/particleman',
      author='Jonathan MacCarthy',
      author_email='jkmacc@lanl.gov',
      install_requires=['numpy', 'matplotlib', 'stockwell>=1.0'],
      python_requires='>=3.4',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
     )
