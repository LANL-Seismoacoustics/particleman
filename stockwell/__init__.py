from stockwell.st import st
import numpy as np

def stransform(X, hp=0, lp=0, Fs=0):
    """Perform a Stockwell transform on a time-series.

    Returns the transform (S), and time (T) and frequency (F)
    matrices suitable for use with the contour/contourf functions.

    Parameters
    ----------
    X : numpy.ndarray
        array containing time-series data
    hp : float
        high-pass point in samples (if Fs is not specified) or in Hz (if Fs is specified)
    lp : float
        low-pass point in samples (if Fs is not specified) or in Hz (if Fs is specified)
    Fs : float
        sampling rate in Hz

    Returns
    -------
    S, T, F : numpy.ndarray, rand 2
        Transform (S), time (T), and frequency (F) matrices.

    Examples
    --------
    # for a 100 Hz time series
    >>> S, T, F = stransform(data, Fs=100)
    >>> plt.contourf(T, F, abs(S))

    References
    ----------
    http://vcs.ynic.york.ac.uk/docs/naf/intro/concepts/timefreq.html
    http://kurage.nimh.nih.gov/meglab/Meg/Stockwell

    """
    L = len(X)
    if Fs:
        # If the sample rate has been specified then
        # we low-pass at the nyquist frequency.
        if not lp:
            lp = Fs/2.0

        # Providing the sample rate also means that the
        # filter parameters are in Hz, so we convert
        # them to the appropriate number of samples
        low = int(np.floor(hp/(Fs/L)))
        high = int(np.ceil(lp/(Fs/L)))
    else:
        # Since we don't have a sampling rate then
        # everything will be expressed in samples
        if not lp:
            lp = L/2.0
        low = int(hp)
        high = int(lp)

    # The stockwell transform
    S = st(X,low,high)

    # Compute our time and frequency matrix with
    # the correct scaling for use with the
    # contour and contourf functions
    if Fs:
        t = 1.0/Fs # Length of one sample
        t = np.arange(L)*t # List of time values
        T, F = np.meshgrid(t, np.arange(hp, lp, (lp-hp)/(1.0*S.shape[0])))
    else:
        t = np.arange(L)
        T, F = np.meshgrid(t, np.arange(int(hp), int(lp), int(lp-hp)/(1.0*S.shape[0])))

    return S,T,F
