# Particle Man

**Particle** **m**otion **an**alysis of seismic surface waves using the Stockwell Transform.


[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![pipeline status](https://git.lanl.gov/ees-geophysics/particleman/badges/master/pipeline.svg)](https://git.lanl.gov/ees-geophysics/particleman/commits/master)

Seismic surface waves can be detected and extracted from three-component
data through particle motion polarization analysis.  Particle Man is a
software package that employs the Stockwell Transform to perform surface wave
filtering/extraction and display for multifrequency analysis.


This package implements the Normalized Inner Product surface wave filtering and extraction methods of
[Meza-Fajardo et al., (2015)](https://pubs.geoscienceworld.org/ssa/bssa/article/105/1/210/323461/identification-and-extraction-of-surface-waves).
The Stockwell transform is implemented in C using FFTW libraries.

![chirp](docs/src/data/chirp.png "Comparison to FFT for a chirp signal")

![filtered retrograde radial](docs/src/data/stransforms_scalar.png "Extracted Retrograde Rayleigh Waves")


## Installation

### Prerequisites

* C compiler
* fftw3

### Dependencies

* Python 3
* NumPy
* Matplotlib

### Using from a Conda environment.yml

The `environment.yml` file will install everything you need into a `particleman`
environment: `conda env create -f environment.yml`.  To install Particle Man into
an existing Conda environment, make sure you've got the dependencies you need installed
and available, and just use `pip` to install it from source.

### Manually on a Mac

`gcc` from XCode should work fine.  Getting fttw3 is easy with [homebrew](http://brew.sh/) (`brew install fftw`)
or with Conda (`conda install -c conda-forge fftw`).

To Particle Man install from source, just:

```bash
pip install .
```