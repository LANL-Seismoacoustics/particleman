"""
Time-frequency filtering for seismic waves.

To estimate propagation azimuth for a prograde/retrograde Rayleigh wavetrain:

1. Make complex component s-transforms, Sn, Se, Sv
2. Apply phase delay to Sv for normal retrograde motion, advance for prograde motion
3. Convert complex transforms into MxNx2 real ("vector") arrays
4. Calculate theta from theta_r and theta_l

To make a normalized inner product (NIP) filter, given azimuth

5. Rotate Sn, Se through theta, into radial (Sr) and transverse (St) component transforms
6. Calculate NIP using the vector Sr and phase-shifted vector Sv
7. Extract surface waves using NIP >= 0.8 (smoothed) as a filter on Sr, St, Sv, and invert to time domain.
   Rayleigh waves will be on the time-domain radial, Love on the time-domain transverse.


References
----------

Meza-Fajardo, K. C., Papageorgiou, A. S., and Semblat, J. F. (2015). Identification and 
Extraction of Surface Waves from Three Component Seismograms Based on the Normalized Inner 
Product. Bulletin of the Seismological Society of America.

"""
import numpy as np

from .util import stransform, istransform


def shift_phase(Sv, polarization='retrograde'):
    """
    Phase-shift an s-transform by the appropriate phase shift for prograde/retrograde motion.

    Shift is done on a complex MxN array by multiplication with either i or -i (imaginary unit).
    This is mostly a reference for how to do/interpret phase shifts, as it's such a simple thing 
    to do outside of a function.

    Parameters
    ----------
    Sv : numpy.ndarray (complex, rank 2)
    polarization : str, {'retrograde', 'prograde', 'linear'}
        'retrograde' will apply a pi/2 phase advance.
        'prograde' or 'linear' will apply a pi/2 phase delay

    Returns
    -------
    numpy.ndarray (real, rank 3)

    References
    ----------
    Pages 5 and 10 from Meta-Fajardo et al. (2015)

    """
    if polarization is 'retrograde':
        # phase advance
        shft = np.array(1j)
    elif polarization in ('prograde', 'linear'):
        # phase delay
        shft = -np.array(1j)
    else:
        raise ValueError("Polarization must be either 'prograde', 'retrograde', or 'linear'")

    return Sv * shft


def estimate_azimuth(Sv, Sn, Se, polarization, xpr):
    """
    Get instantaneous propagation angle [degrees], under the Rayleigh wave assumption.

    Parameters
    ----------
    Sv, Sn, Se : numpy.ndarray (complex, rank 2)
        The vertical, North, and East component equal-sized complex s-transforms.
    polarization : str, {'retrograde', 'prograde', 'linear'}
        'retrograde' will apply a pi/2 phase advance.
        'prograde' or 'linear' will apply a pi/2 phase delay
    xpr : int
        Sense of propagation.  1 for eastward, -1 for westward.
        Try -int(np.sign(np.sin(np.radians(baz)))), unless they're directly N-S from each other.

    Returns
    -------
    az : numpy.ndarray (real, rank 2)
        Instantaneous Rayleigh wave propagation angle [degrees]

    References
    ----------
    Equations (19), (20), and (21) from Meza-Fajardo et al. (2015)

    """
    Svhat = shift_phase(Sv, polarization)

    num = (Se.real * Svhat.real) + (Se.imag*Svhat.imag)
    denom = (Sn.real * Svhat.real) + (Sn.imag*Svhat.imag)
    theta_r = np.arctan(num/denom)

    theta_I = theta_r + np.pi*(1 - np.sign(np.sin(theta_r))) + \
        np.pi*(1 - np.sign(np.cos(theta_r))) * np.sign(np.sign(theta_r))/2

    theta = theta_I + (np.pi/2)*(np.sign(np.sin(theta_I)) - np.sign(xpr))

    return np.degrees(theta)


def rotate_NE_RT(Sn, Se, az):
    """
    Rotate North and East s-transforms to radial and transverse, through an angle.

    Parameters
    ----------
    Sn, Se : numpy.ndarray (complex, rank 2)
        Complex, equal-sized s-transform arrays, for North and East components, respectively.
    az : float
        Rotation angle [degrees].

    Returns
    -------
    Sr, St : numpy.ndarray (rank 2)
        Complex s-transform arrays for radial and transverse components, respectively.

    References
    ----------
    Equation (17) from Meta-Fajardo et al. (2015)

    """
    theta = np.radians(az)

    Sr = np.cos(theta)*Sn + np.sin(theta)*Se
    St = -np.sin(theta)*Sn + np.cos(theta)*Se

    return Sr, St


def NIP(Sr, Sv, polarization=None, eps=None):
    """
    Get the normalized inner product of two complex MxN stockwell transforms.

    Parameters
    ----------
    Sr, Sv: numpy.ndarray (complex, rank 2)
        The radial and vertical component s-transforms.
    polarization : str, optional
        If provided, the Sv will be phase-shifted according to this string before calculating the NIP.
        'retrograde' will apply a pi/2 phase advance (1j * Sv)
        'prograde' or 'linear' will apply a pi/2 phase delay (-1j * Sv)
        If omitted, Sv is assumed to already be phase-shifted according to the desired polarization.
    eps : float, optional
        Tolerance for small denominator values, for numerical stability.
        Useful for synthetic noise-free data.  Authors used 0.04.

    Returns
    -------
    nip : numpy.ndarray (rank 2)
        MxN array of floats between -1 and 1.

    References
    ----------
    Equation (16) and (26) from Meza-Fajardo et al. (2015)

    """
    if polarization:
        Svhat = shift_phase(Sv, polarization)
    else:
        Svhat = Sv

    Avhat = np.abs(Svhat)
    if eps:
        mask = (Avhat / Avhat.max()) < eps
        Avhat[mask] += eps*Avhat.max()

    ip = Sr.real*Svhat.real + Sr.imag*Svhat.imag
    n = np.abs(Sr) * Avhat

    return ip/n


def smooth_NIP(X, corner=None):
    """
    Smooth any sharp edges in the NIP.

    Uses a hyperbolic tangent, with provided critical value.

    References
    ----------
    Maceira and Ammon (2009)

    """
    phi = (np.pi/2)*(1.0+np.tanh((X-corner)))/2

    return np.sin(phi)**2 * NIP


def get_filter(nip, polarization, threshold=0.8, width=0.1):
    """
    Get an NIP-based filter that will pass waves of the specified type.
    
    The filter is made from the NIP and cosine taper for the specified wave type.
    The nip and the polarization type must match.

    Parameters
    ----------
    nip : numpy.ndarray (real, rank 2)
        The NIP array [-1.0, 1.0]
    polarization : str
        The type of polarization that was used to calculate the provided NIP.
        'retrograde', 'prograde', or 'linear'.  See "NIP" function.
    threshold, width : float
        The cosine taper critical/crossover value ("x_r") and width ("\Delta x")

    Returns
    -------
    numpy.ndarray (real, rank 2)
        The NIP-based filter array [0.0, 1.0] to multiply into the complex Stockwell arrays,
        before inverse transforming to the time-domain.

    References
    ----------
    Equation (27) and (28) from Meza-Fajardo et al. (2015)

    """
    if polarization is 'retrograde':
        filt = np.zeros(nip.shape)
        mid = (threshold - width < nip) & (nip < threshold)
        high = threshold < nip
        filt[mid] = 0.5 * np.cos((np.pi*(nip[mid]-threshold))/width) + 0.5
        filt[high] = 1.0
    elif polarization in ('prograde', 'linear'):
        filt = np.ones(nip.shape)
        mid = (threshold < nip) & (nip < threshold + width)
        high = threshold + width < nip 
        filt[mid] = 0.5 * np.cos((np.pi*(nip[mid]-threshold))/width) + 0.5
        filt[high] = 0.0
    else:
        raise ValueError('Unknown polarization type.')

    return filt


def NIP_filter(n, e, z, fs, xpr, polarization='retrograde', threshold=0.8, width=0.1, eps=None):
    """
    Filter a 3-component seismogram based on the NIP criterion.

    This is a composite convenience routine that uses sane defaults.
    If you want to get intermediate products, call internal routines individually.

    Parameters
    ----------
    n, e, z : numpy.ndarray (rank 1)
        Equal-length data arrays for North, East, and Vertical components, respectively.
    fs : float
        Sampling frequency [Hz]
    xpr : int
        Sense of wave propagation. -1 for westward, 1 for eastward.
        Try np.sign(station_lon - event_lon), unless they're directly N-S from each other.
    polarization : str
        'retrograde' to extract normal retrograde Rayleigh waves
        'prograde' to extract prograde Rayleigh waves
        'linear' to extract Love waves
    threshold, width : float
        The critical value ("x_r") and width ("\Delta x") for the NIP filter (cosine) taper.
    eps : float
        Tolerance for small NIP denominator values, for numerical stability.

    Returns
    -------
    r, t, v : numpy.ndarray (rank 1)
        Filtered radial, traverse, and vertical components.
    theta : float
        angle [degrees].

    """
    Sn = stransform(n, Fs=fs)
    Se = stransform(e, Fs=fs)
    Sv = stransform(z, Fs=fs)
    
    #Svhat = shift_phase(Sv, polarization=polarization)

    theta = estimate_azimuth(Sv, Sn, Se, polarization, xpr)

    Sr, St = rotate_NE_RT(Sn, Se, theta)

    nip = NIP(Sr, Sv, polarization)

    # get the filter
    filt = get_filter(nip, threshold=0.8, width=0.1, polarization=polarization)

    # apply the filter
    Srf = Sr*filt
    Stf = St*filt

    # XXX: damp any nans?
    Srf[np.isnan(Srf)] = 0.0
    Srt[np.isnan(Srt)] = 0.0

    # return to time domain
    r = istransform(Srf, Fs=fs)
    t = istransform(Stf, Fs=fs)
    
    return r, t

