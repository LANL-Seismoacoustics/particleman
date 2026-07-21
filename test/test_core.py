"""
Test core stockwell functionality.

"""
import os

import numpy as np
from numpy.fft import fft
import pytest

from particleman import stransform, istransform
from particleman.st import st, ist

def test_stransform(synthetic_data):
    """
    The forward/inverse S transform roundtrip gives me back what I started with.
    """
    az_retro, n, e, v, fs = synthetic_data

    Sn = stransform(n, Fs=fs)
    n_roundtrip = istransform(Sn, Fs=fs)

    assert np.allclose(n_roundtrip, n)


def test_default_frequency_range_bug():
    """
    Document that the Python wrapper for st() has problematic default handling.
    
    The Python st() function has this code:
        if (lo is None) and (hi is None):
            hi = N % 2  # BUG! Should be N // 2
    
    However, stransform() works around this by explicitly passing lo and hi values,
    so user-facing code is not affected by this bug.
    
    After migration to external stockwell library, the low-level st() will have
    correct default handling: hi = N // 2
    """
    # This test just documents that the bug exists in the code
    # We can't easily trigger it without causing segfaults
    # The stransform() wrapper protects users from this bug
    
    # stransform works correctly because it always passes explicit values:
    N = 100
    data = np.sin(2 * np.pi * 10 * np.arange(N) / N)
    S = stransform(data)  # This works fine
    
    # Expected: N//2 + 1 = 51 frequency bins (0 to 50 Hz)
    assert S.shape[0] == 51
    assert S.shape[1] == N


def test_gamma_parameter():
    """
    Test that the gamma parameter affects the transform.
    
    Gamma controls the time-frequency resolution tradeoff:
    - gamma < 1: Better time resolution
    - gamma = 1: Standard (default)
    - gamma > 1: Better frequency resolution
    """
    N = 100
    data = np.sin(2 * np.pi * 10 * np.arange(N) / N)
    
    # Different gamma values should produce different transforms
    S_low = stransform(data, gamma=0.5)
    S_default = stransform(data, gamma=1.0)
    S_high = stransform(data, gamma=2.0)
    
    # All should have the same shape
    assert S_low.shape == S_default.shape == S_high.shape
    
    # But different content
    assert not np.allclose(S_low, S_high)
    assert not np.allclose(S_low, S_default)
    assert not np.allclose(S_default, S_high)


def test_window_types():
    """
    Test that different window types produce different transforms.
    
    Two window types are available:
    - 'gauss': Standard Gaussian window (default)
    - 'kazemi': Alternative window from Kazemi (2014)
    """
    N = 100
    data = np.sin(2 * np.pi * 10 * np.arange(N) / N)
    
    S_gauss = stransform(data, win_type='gauss')
    S_kazemi = stransform(data, win_type='kazemi')
    
    # Same shape
    assert S_gauss.shape == S_kazemi.shape
    
    # Different results
    assert not np.allclose(S_gauss, S_kazemi)


def test_fft():
    """
    The S transform is the same as a time-integrated FFT.
    """
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
