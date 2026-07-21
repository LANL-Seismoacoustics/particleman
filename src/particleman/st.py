"""
Backward-compatible wrapper around external stockwell library.

This module provides the same API as the old internal st implementation,
but uses the external stockwell library which fixes bugs and adds features.

.. deprecated:: 0.4.0
    The functions in this module (st, ist) are deprecated.
    Use the stockwell library directly instead:
    
    Instead of::
    
        from particleman.st import st, ist
    
    Use::
    
        from stockwell.st import st, ist
    
    The high-level functions particleman.stransform() and particleman.istransform()
    are NOT deprecated and should continue to be used.
"""
import warnings
import numpy as np
from stockwell.st import st as _st, ist as _ist


def st(data, lo=None, hi=None, gamma=1.0, win_type='gauss'):
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

    Parameters
    ----------
    data : array_like
        Input data array.
    lo : int, optional
        Lowest frequency index to return (default 0).
    hi : int, optional
        Highest frequency index to return (default n/2), where n is the
        length of data.
    gamma : float, optional
        Gamma parameter (default 1). Controls time-frequency resolution.
        Higher values increase frequency resolution at expense of time resolution.
    win_type : {'gauss', 'kazemi'}, optional
        Window type (default 'gauss').

    Returns
    -------
    result : ndarray
        The Stockwell transform of data. Shape: (nfreqs, ntimes)

    Notes
    -----
    .. deprecated:: 0.4.0
        `particleman.st.st()` is deprecated. Use `stockwell.st.st()` directly instead.
        This wrapper is maintained for backward compatibility but will be removed
        in a future version.
    
    The gamma parameter can be used to tune resolution:
    - gamma < 1: Better time resolution, worse frequency resolution
    - gamma = 1: Standard Stockwell transform (default)
    - gamma > 1: Better frequency resolution, worse time resolution

    Two window types are available:
    - 'gauss' (default): Standard Gaussian window
    - 'kazemi': Alternative window from Kazemi (2014)
    """
    warnings.warn(
        "particleman.st.st() is deprecated and will be removed in a future version. "
        "Use 'from stockwell.st import st' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Handle backward compatibility with None defaults
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(data) // 2  # Fixed: was N % 2 (bug), now N // 2 (correct)
    
    # Call external stockwell library
    return _st(data, lo=lo, hi=hi, gamma=gamma, win_type=win_type)


def ist(X, lo=None, hi=None):
    """
    Inverse Stockwell transform.

    Parameters
    ----------
    X : array_like
        Input data array, 2d and complex.
    lo : int, optional
        Lowest frequency index to use (default 0).
    hi : int, optional
        Highest frequency index to use (default n/2), where n is the number of
        time samples in X.

    Returns
    -------
    result : ndarray
        The inverse Stockwell transform of X.

    Notes
    -----
    .. deprecated:: 0.4.0
        `particleman.st.ist()` is deprecated. Use `stockwell.st.ist()` directly instead.
        This wrapper is maintained for backward compatibility but will be removed
        in a future version.
    
    The difference between hi and lo must be equal to the number of
    frequencies in X (first dimension) minus 1.
    """
    warnings.warn(
        "particleman.st.ist() is deprecated and will be removed in a future version. "
        "Use 'from stockwell.st import ist' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    X = np.ascontiguousarray(X, dtype=np.complex128)

    N, M = X.shape

    # Handle backward compatibility with None defaults
    if lo is None:
        lo = 0
    if hi is None:
        hi = M // 2  # Fixed: was M % 2 (bug), now M // 2 (correct)

    # Validate dimensions
    if hi - lo + 1 != N:
        raise ValueError("Inconsistent dimensions")

    # Call external stockwell library
    return _ist(X, lo=lo, hi=hi)


# Export the functions
__all__ = ['st', 'ist']
