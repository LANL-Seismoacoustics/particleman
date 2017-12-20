"""
Test core stockwell functionality.

"""
import os

import numpy as np
from stockwell import stransform, istransform

here = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.sep.join([here, 'data', 'BW.RJOB..EHZ.txt']))

def test_stransform():
    S = stransform(data, Fs=100)
    s = istransform(S, Fs=100)
    assert np.allclose(s, data)
