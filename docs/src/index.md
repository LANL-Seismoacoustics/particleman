# **P**article **Man**

**Particle** **m**otion **an**alysis of seismic surface waves using the Stockwell Transform.

Seismic surface waves can be detected and extracted from three-component
data through particle motion polarization analysis.  Particle Man is a
software package that employs the Stockwell Transform to perform surface wave
filtering/extraction and display for multifrequency analysis.



The [Stockwell transform](https://en.wikipedia.org/wiki/S_transform) is like a
multi-scale spectrogram.  It uses variable-width gaussian windows instead of a
constant boxcar window to do localize the spectrum in time.  


Features: 

* The time-frequency tile can be directly interpreted like a spectrogram, unlike
  a wavelet transform "scalogram".
* Multi-scale time-frequency phase analysis can be performed directly on the
  S-transforms (see [filtering](filtering.md)).
* No *a priori* frequency resolution constraints are imposed on the transform,
  unlike a spectrogram, which requires a boxcar window length.

![chirp](data/chirp.png "chirp")



## Installation

Particle Man has been tested on a Mac.  I suspect Linux will also work.


### Dependencies

* Python 2.7
* NumPy
* Matplotlib
* C compiler
* fftw3


On a Mac, gcc from XCode should work fine.  Getting fttw3 is easy with [homebrew](http://brew.sh/).
To install, just:

```bash
git clone https://git.lanl.gov/ees-geophysics/pystockwell.git
cd pystockwell
pip install .
```

...or, directly from the repository:

```bash
pip install git+https://git.lanl.gov/ees-geophysics/pystockwell.git
```

### Credits

All credit goes to the original authors.  Please see LICENSE.txt.

This package is an implementation of the [fast
S-transform](http://ieeexplore.ieee.org/document/4649729/) originally in the
National Institutes of Mental Health [NIMH
MEG](http://kurage.nimh.nih.gov/meglab/Meg/Stockwell) project and University of
York, York Neuroimaging Centre [NeuroImaging Analysis
Framework](http://vcs.ynic.york.ac.uk/docs/naf/index.html) (see their
[Time-Frequency Analyses
section](http://vcs.ynic.york.ac.uk/docs/naf/intro/concepts/timefreq.html)).

This package offers new plotting and filtering functionality.




