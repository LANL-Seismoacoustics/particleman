import numpy as np
import pytest
from particleman import stransform

@pytest.fixture
def synthetic_data():
    """
    Follows Meza-Fajardo et al. 2015 "Extraction of Retrograde Rayleigh Wave"
    synthetic example, and figure 4.

    """
    fs = 20 # sampling frequency
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
    az_retro = 60
    theta = np.radians(az_retro)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    n, e = R @ np.array([y, x])
    v = z

    return az_retro, n, e, v, fs