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

Meza‐Fajardo, K. C., Papageorgiou, A. S., & Semblat, J. F. (2015). Identification and 
Extraction of Surface Waves from Three‐Component Seismograms Based on the Normalized Inner 
Product. Bulletin of the Seismological Society of America.

"""
from .util import stransform, istransform

def _complex_to_vector(S):
    """
    Turn the complex MxN numpy array S into an MxNx2 matrix with the first page being the real part,
    and the second page being the imaginary part.

    Parameters
    ----------
    S : numpy.ndarray (complex, rank 2)

    Returns
    -------
    numpy.ndarray (real, rank 3)

    References
    ----------
    Equation (14) from Meza‐Fajardo et al. (2015)

    """
    V = np.empty(S.shape + (2,))
    V[:,:,0] = S.real
    V[:,:,1] = S.imag

    return V


def shift_phase(S, motion='retrograde'):
    """
    Phase-shift a MxN s-transform by the appropriate phase shift for prograde/retrograde motion.

    Parameters
    ----------
    V : numpy.ndarray (real, rank 3)

    Returns
    -------
    numpy.ndarray (real, rank 3)

    References
    ----------
    Page 5 from Meza‐Fajardo et al. (2015)

    """
    #XXX: check this
    if motion is 'retrograde':
        ph = 1j
    else:
        ph = -1j

    return S * ph


def rayleigh_azimuth(Svhat, Sn, Se, xpr):
    """
    Get instantaneous propagation angle [degrees], under the Rayleigh wave assumption.

    Parameters
    ----------
    Sv, Sn, Se : numpy.ndarray (complex, rank 2)
        The vertical, North, and East component equal-sized complex s-transforms.
    xpr : int
        Sense of propagation.  1 for eastward, -1 for westward.
        Try np.sign(station_lon - event_lon), unless they're directly N-S from each other.

    Returns
    -------
    az : numpy.ndarray (real, rank 2)
        Instantaneous Rayleigh wave propagation angle [degrees]

    References
    ----------
    Equations (19), (20), and (21) from Meza‐Fajardo et al. (2015)

    """
    #TODO: is these even necessary?
    #Vhat = _complex_to_vector(Svhat)
    #E = _complex_to_vector(Se)
    #N = _complex_to_vector(Sn)

    num = (Se.real * Svhat.real) + (Se.imag*Svhat.imag)
    denom = (Sn.real * Svhat.real) + (Sn.imag*Svhat.imag)
    theta_r = np.arctan(num/denom)

    theta_I = theta_r + np.pi*(1 - np.sign(np.sin(theta_r)))
        + np.pi*(1 - np.sign(np.cos(theta_r))) * np.sign(np.sign(theta_r))/2

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
    Equation (17) from Meza‐Fajardo et al. (2015)

    """
    theta = np.radians(az)
    Sr = np.cos(theta)*Sn + np.sin(theta)*Se
    St = -np.sin(theta)*Sn + np.cos(theta)*Se

    return Sr, St


def NIP(Sr, Svhat, eps=None):
    """
    Get the normalized inner product of two complex MxN stockwell transforms.

    Parameters
    ----------
    R, Vhat : numpy.ndarray (complex, rank 2)
        The radial and shifted vertical component s-transforms.
    eps : float, optional
        Tolerance for small denominator values, for numerical stability (e.g. noise-free data)
        Authors used 0.04.

    Returns
    -------
    nip : numpy.ndarray (rank 2)
        MxN array of floats between -1 and 1.

    References
    ----------
    Equation (16) and (26) from Meza‐Fajardo et al. (2015)

    """
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
    #x = np.linspace(-1.0, 1.0, X.size, endpoint=True)
    phi = (np.pi/2)*(1.0+np.tanh((X-corner)))/2

    return np.sin(phi)**2 * NIP


def NIP_filter(n, e, z, fs, xpr, motion='retrograde', threshold=0.8, eps=0.04):
    """
    Filter a 3-component seismogram based on the NIP criterion.

    This is a composite convenience routine that uses sane defaults.
    If you want to get intermediate products, call internal routines individually.

    Parameters
    ----------
    n, e, z : numpy.ndarray (rank 1)
        Equal-length data arrays for North, East, and Vertical components, respectively.
    xpr : int
        Sense of wave propagation. -1 for westward, 1 for eastward.
        Try np.sign(station_lon - event_lon), unless they're directly N-S from each other.
    fs : float
        Sampling frequency [Hz]
    threshold : float
        NIP (soft) cutoff value for filter design.
    eps : float
        Tolerance for small NIP denominator values, for numerical stability.

    Returns
    -------
    r, t, v : numpy.ndarray (rank 1)
        Filtered radial, traverse, and vertical components.
    theta :
        Rotation angle.

    """
    Sn, _, _ = stransform(n, Fs=fs)
    Se, _, _ = stransform(e, Fs=fs)
    Sv, _, _ = stransform(z, Fs=fs)
    
    #Svhat = shift_phase(Sv, motion='retrograde')
    if motion is 'retrograde':
        Svhat = Sv * 1j
    elif motion is 'prograde':
        Svhat = Sv * -1j
    else:
        raise ValueError('Unknown motion type')

    theta = rayleigh_azimuth(Svhat, Sn, Se, xpr)

    Sr, St = rotate_NE_RT(Sn, Se, theta)

    nip = NIP(Sr, Svhat)

    #filt = smooth_array(nip, corner=0.8)

    Sr = np.where(nip >= 0.8, Sr, 0.0)
    St = np.where(nip >= 0.8, St, 0.0)
    #Sv = np.where(nip >= 0.8, Sv, 0.0)

    return istransform(Sr, Fs=fs), istransform(St, Fs=fs)

    
    


