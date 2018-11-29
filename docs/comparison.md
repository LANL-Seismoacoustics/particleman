## Comparison to Spectrogram 

The Stockwell transform retains both broad-scale and fine-scale structure,
compared to a spectrogram, whose frequency resolution is is limited by the
moving window size chosen.


![waveform](data/RJOB_EHZ_wave.png)

**Spectrogram**

![spectrogram](data/spectrogram.png)


## Comparison to Morlet Transform

Wavelet transforms like the Morlet transform, for example, also do this, but
the y-dimension of the 'scalogram' is not frequency, it is "scale".

![Morlet comparison](data/morlet_comparison.png)
