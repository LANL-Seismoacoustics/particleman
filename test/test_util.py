"""
Test core stockwell functionality.

"""
import os

import numpy as np
from numpy.fft import fft
from stockwell import stransform, istransform

here = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.sep.join([here, 'data', 'BW.RJOB..EHZ.txt']))

def test_stransform():
    S = stransform(data, Fs=100)
    s = istransform(S, Fs=100)
    assert np.allclose(s, data)

def test_fft():
    sample_rate = 40.0  # [Hz]
    total_sec = 30.0
    t = np.arange(0.0, total_sec, 1.0 / sample_rate)

    # make a signal
    c = np.cos(2 * np.pi * sample_rate * t)

    N = len(c)
    fft_c = np.abs(fft(c, N) * sample_rate)[:int(N/2) + 1] * 1.0 / N

    S = stransform(c, Fs=sample_rate, lp=sample_rate / 2, hp=0)
    fft_S = abs(S.sum(axis=1)) / (t[-1] - t[0])

    # XXX: this tolerance is too big.
    assert np.allclose(fft_S, fft_c, atol=0.1)
