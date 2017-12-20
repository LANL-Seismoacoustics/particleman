# Stockwell Transform

The Stockwell transform, and its inverse, implemented in C and NumPy.

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)


## Example

### Transform a chirp.

```python
import numpy as np
from scipy.signal import chirp
from stockwell import stransform

sample_rate = 40.0  #[Hz]
total_sec = 30.0

# make a linear chirp
t = np.arange(0.,total_sec,1./sample_rate)
c = chirp(t, 0.2, 20.0, 10.0, method='linear', phi=0, vertex_zero=True)

S, T, F = stransform(c, Fs=sample_rate)

```

### Plotting

Plotting is easy.

```python
plt.pcolormesh(T, F, abs(S))

```

Let's plot something nicer, though.

```python
N = len(c)
C = np.abs(np.fft.fft(c, N) * sample_rate)[:N/2 + 1] * 2.0/N
df = 1.0 / (N/sample_rate)
f = df * np.arange(len(C))
dt = 1.0 / sample_rate

fig = plt.figure(figsize=(15,7))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax1.plot(t, c)
ax1.set_title('chirp')
ax1.set_ylabel('amplitude')

ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax2.pcolormesh(T, F, abs(S))
ax2.set_title('s-transform, S(t,f)')
ax2.set_xlabel('time [sec]')
ax2.set_ylabel('frequency [Hz]')

ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax3.plot(abs(S.sum(axis=1))/(t[-1] - t[0]), F[:,0], 'r', label=r'$\sum S(t,f) \Delta t$')
ax3.plot(C, f, '--k', label='FFT')
ax3.set_title('S(t,f) vs FFT')
ax3.set_xlabel('amplitude')
plt.legend()

plt.tight_layout()
plt.draw()

``` 

![chirp](docs/src/data/chirp.png "chirp")


### Advanced Filtering

One of the benefits of the Stockwell transform is that it preserves phase in the
same way that a Fourier transform does, which makes filters based upon phase
relationships possible in time-frequency space.  In the example below, we use
the Normalized Inner Produce filter of Meza-Fajardo et al., (2015) to pass
retrograde Rayleigh waves (in any frequency) from a particular azimuth.

![filtered retrograde radial](docs/src/data/stransforms_scalar.png)


## Installation

### Dependencies

* NumPy
* gcc
* fftw3

### Using Conda

The `environment.yml` file will install everything you need into a `pystockwell`
environment: `conda env create -f environment.yml`.  To install PyStockwell into
an existing Conda environment, make sure you've got the dependencies you need installed
and available, and just use `pip` to install it from source.

### Manually on a Mac

`gcc` from XCode should work fine.  Getting fttw3 is easy with [homebrew](http://brew.sh/) (`brew install fftw`)
or with Conda (`conda install -c conda-forge fftw`).

To PyStockwell install from source, just:

```bash
pip install .
```

...or from the remote repo:

```bash
pip install git+https://git.lanl.gov/ees-geophysics/pystockwell.git
```

