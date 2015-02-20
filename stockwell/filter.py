"""
Time-frequency filtering for seismic waves.

To estimate propagation azimuth for a prograde/retrograde Rayleigh wavetrain:

1. Make complex component s-transforms, Sn, Se, Sv
2. Apply phase delay Sv for normal retrograde motion, advance for prograde motion
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
    pass


def phase_shift(V, motion='retrograde'):
    """
    Phase-shift a MxNx2 s-transform by the appropriate phase shift for prograde/retrograde motion.

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
    pass


def rayleigh_azimuth(Sv, Sn, Se):
    """
    Get instantaneous propagation angle [degrees], under the Rayleigh wave assumption.

    Parameters
    ----------
    Sv, Sn, Se : numpy.ndarray (complex, rank 2)
        The vertical, North, and East component equal-sized complex s-transforms.

    Returns
    -------
    az : numpy.ndarray (real, rank 2)
        Instantaneous Rayleigh wave propagation angle [degrees]

    References
    ----------
    Equations (18) and (19) from Meza‐Fajardo et al. (2015)

    """
    pass


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
    pass


def NIP(R, Vhat, eps=0.04):
    """
    Get the normalized inner product of two MxNx2 arrays.

    Parameters
    ----------
    R, Vhat : numpy.ndarray (real, rank 3)
        The radial and shifted vertical component s-transforms.
    eps : float
        Tolerance for small denominator values, for numerical stability.

    Returns
    -------
    nip : numpy.ndarray (rank 2)
        MxN array of floats between 0 and 1.

    References
    ----------
    Equation (16) and (26) from Meza‐Fajardo et al. (2015)

    """
    pass


def smooth_array(X):
    """
    Smooth any sharp edges in the NIP.

    """
    pass


def NIP_filter(n, e, v, threshold=0.8, eps=0.04):
    """
    Filter a 3-component seismogram based on the NIP criterion.

    This is a convenience routine that hides the messiness of the procedure and uses sane defaults.
    If you want to get intermediate products, call the routines individually.

    Parameters
    ----------
    n, e, v : numpy.ndarray (rank 1)
        Equal-length data arrays for North, East, and Vertical components, respectively.
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
    pass
