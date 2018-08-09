"""
Test core stockwell functionality.

"""
import os

import numpy as np
from numpy.fft import fft
from particleman import stransform, istransform

here = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.sep.join([here, 'data', 'BW.RJOB..EHZ.txt']))

def make_test_data():
    """
    Follows Meza-Fajardo et al. 2015 synthetic data.
    """
    fs = 40 # sampling frequency
    dt = 1 / fs
    t = np.arange(0, 20, dt)

    taper = np.zeros_like(t)
    hanning_idx = np.flatnonzero(np.logical_and(5 < t, t < 15))
    taper[hanning_idx] = np.hanning(len(hanning_idx))

    S5 = 0.6 * np.sin(5 * 2*np.pi * t) * taper
    S2 = 1 * np.sin(2 * 2*np.pi * t) * taper
    S1 = 1 * np.sin(1 * 2*np.pi * t) * taper

    # pi/2 phase delay S2
    S2_minus = 1 * np.sin(2 * 2*np.pi * t - np.pi/2) * taper
    # pi/2 phase advanced S1
    S1_plus = 1 * np.sin(1 * 2*np.pi * t + np.pi/2) * taper

    x = (1 / np.sqrt(2))*S5 + 0.5*S1
    y = (1 / np.sqrt(2))*S5 + 0.5*S2
    z =  S1_plus + S2_minus

    # rotate through rotation matrix
    az = 60
    theta = np.radians(az)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    N, E = R @ np.array([y, x])
    V = z

    return t, N, E, V


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
