""" ctypes interface to st.c
"""
import ctypes
from distutils import sysconfig
import os

import numpy as np

ext, = sysconfig.get_config_vars('SO')
libst = ctypes.CDLL(os.path.dirname(__file__) + '/src/st' + ext)

# void st(int len, int lo, int hi, double *data, double *result)
libst.st.restype = None
libst.st.argtypes = [ctypes.c_int, ctypes.c_int,
                     ctypes.pointer(ctypes.c_double),
                     ctypes.pointer(ctypes.c_double)]


def st(data, lo=None, hi=None):
    """
    st(x[, lo, hi]) returns the 2d, complex Stockwell transform of the real
    array x. If lo and hi are specified, only those frequencies (rows) are
    returned; lo and hi default to 0 and n/2, resp., where n is the length of x.

    Stockwell transform of the real array data. The number of time points need
    not be a power of two. The lo and hi arguments specify the range of
    frequencies to return, in Hz. If they are both zero, they default to lo = 0
    and hi = len / 2. The result is returned in the complex array result, which
    must be preallocated, with n rows and len columns, where n is hi - lo + 1.
    For the default values of lo and hi, n is len / 2 + 1.

    """
    N = data.shape[0]

    if (lo is None) and (hi is None):
        # uses C division, following the old stmodule.c
        hi = divmod(N / 2)

    M = hi - lo + 1

    data = np.ascontiguousarray(data, dtype=np.double)
    results = np.empty((N, M), dtype=np.complex)

    # void st(int len, int lo, int hi, double *data, double *result)
    libst.st(N, lo, hi, data.ctypes.data_as(ctypes.pointer(ctypes.c_double)),
             results.ctypes.data_as(ctypes.pointer(ctypes.c_double)))

    return results

def ist():
    pass