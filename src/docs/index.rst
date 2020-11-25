.. particleman documentation master file, created by
   sphinx-quickstart on Mon Aug 27 15:50:14 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to particleman's documentation!
=======================================

**Particle m**\ otion **an**\ alysis of seismic surface waves using the Stockwell Transform.

Seismic surface waves can be detected and extracted from three-component
data through particle motion polarization analysis.  Particle Man is a
software package that employs the Stockwell Transform to perform surface wave
filtering/extraction and display for multifrequency analysis.

Features: 

* The time-frequency tile can be directly interpreted like a spectrogram, unlike
  a wavelet transform "scalogram".
* Multi-scale time-frequency phase analysis can be performed directly on the
  S-transforms.
* No *a priori* frequency resolution constraints are imposed on the transform,
  unlike a spectrogram, which requires a boxcar window length.


.. toctree::
   :maxdepth: 2
   :hidden:

   basic
   comparison
   filtering
   plotting
   API Reference <api/modules.rst>
   license
   changelog
   Repository <https://github.com/LANL-Seismoacoustics/particleman>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
