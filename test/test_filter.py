import numpy as np
import pytest
from particleman import stransform, istransform
import particleman.filter as filt

idx_7_5hz = 150
idx_5hz = 100
idx_2hz = 40
idx_1hz = 20
idx_2sec = 40
idx_10sec = 200

def test_xpr():
    assert filt.xpr(45) == 1
    assert filt.xpr(160) == 1
    assert filt.xpr(220) == -1
    assert filt.xpr(259) == -1


def test_instantaneous_azimuth(synthetic_data):
    """
    The instantaneous azimuth at 1 Hz should be ~150
    The instantaneous azimuth at 2 Hz should be ~60
    The instantaneous azimuth at 5 Hz should be ~105
    """
    TOL = 1.0 # [degrees]
    POLARIZATION = 'retrograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)

    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)

    assert theta[idx_5hz, idx_10sec] == pytest.approx(az_retro + 45, abs=TOL) # 60 (y axis retrograde wave) + 45 (x-y linear wave)
    assert theta[idx_2hz, idx_10sec] == pytest.approx(az_retro, abs=TOL)
    assert theta[idx_1hz, idx_10sec] == pytest.approx(az_retro + 90, abs=TOL) # 60 (retrograde wave) + 90 
    assert theta[idx_7_5hz, idx_2sec] == pytest.approx(az_retro + 90, abs=TOL) # 60 (retrograde wave) + 90 


def test_rotate_NE_RT(synthetic_data):
    """
    After rotation, the transverse component should contain almost no amplitude,
    and the radial component should contain all of it.
    """
    TOL = 0.01 # signal amplitude
    POLARIZATION = 'retrograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)

    assert abs(Sr)[idx_5hz, idx_10sec] == pytest.approx(0.6, abs=TOL)
    assert abs(Sr)[idx_1hz, idx_10sec] > 0.4  # this is number somewhat arbitrary.  should go through the equations to figure out what it should be.
    assert abs(Sr)[idx_2hz, idx_10sec] > 0.4  # this is number somewhat arbitrary.  should go through the equations to figure out what it should be.

    assert abs(St)[idx_5hz, idx_10sec] == pytest.approx(0.0)
    assert abs(St)[idx_2hz, idx_10sec] == pytest.approx(0.0)
    assert abs(St)[idx_1hz, idx_10sec] == pytest.approx(0.0)


def test_NIP_retrograde(synthetic_data):
    TOL = 0.05
    POLARIZATION = 'retrograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
    NIP = filt.NIP(Sr, Sv, polarization=POLARIZATION, eps=0.04)

    assert NIP[idx_1hz, idx_10sec] == pytest.approx(-1.0, abs=TOL)
    assert NIP[idx_2hz, idx_10sec] == pytest.approx(1.0, abs=TOL)
    assert NIP[idx_5hz, idx_10sec] == pytest.approx(0.0, abs=TOL)


def test_get_filter_retrograde(synthetic_data):
    TOL = 0.05
    POLARIZATION = 'retrograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
    NIP = filt.NIP(Sr, Sv, polarization=POLARIZATION, eps=0.04)

    f = filt.get_filter(NIP, POLARIZATION, threshold=0.8, width=0.1)
    Srf = Sr * f
    Svf = Sv * f

    assert abs(Srf[idx_5hz, idx_10sec]) == pytest.approx(0.0, abs=TOL)
    assert abs(Srf[idx_2hz, idx_10sec]) == pytest.approx(0.5, abs=TOL)
    assert abs(Srf[idx_1hz, idx_10sec]) == pytest.approx(0.0, abs=TOL)

    rf = istransform(Sr * f, Fs=fs)
    tf = istransform(St * f, Fs=fs)
    vf = istransform(Sv * f, Fs=fs)

    assert tf.max() == pytest.approx(0.0, abs=TOL)
    assert rf.max() == pytest.approx(0.5, abs=TOL)
    assert vf.max() == pytest.approx(1.0, abs=TOL)


def test_scalar_azimuth_retrograde(synthetic_data):
    TOL = 1.0 
    POLARIZATION = 'retrograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
    NIP = filt.NIP(Sr, Sv, polarization=POLARIZATION, eps=0.04)

    f = filt.get_filter(NIP, POLARIZATION, threshold=0.8, width=0.1)
    Svhat = filt.shift_phase(Sv, POLARIZATION)

    ef = istransform(Se * f, Fs=fs)
    nf = istransform(Sn * f, Fs=fs)
    vfhat = istransform(Svhat * f, Fs=fs)

    az = filt.scalar_azimuth(ef, nf, vfhat)

    assert az == pytest.approx(az_retro, abs=TOL)


def test_NIP_prograde(synthetic_data):
    TOL = 0.05
    POLARIZATION = 'prograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
    NIP = filt.NIP(Sr, Sv, polarization=POLARIZATION, eps=0.04)

    assert NIP[idx_1hz, idx_10sec] == pytest.approx(1.0, abs=TOL)
    assert NIP[idx_2hz, idx_10sec] == pytest.approx(-1.0, abs=TOL)
    assert NIP[idx_5hz, idx_10sec] == pytest.approx(0.0, abs=TOL)


def test_get_filter_prograde(synthetic_data):
    TOL = 0.05
    POLARIZATION = 'prograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
    NIP = filt.NIP(Sr, Sv, polarization=POLARIZATION, eps=0.04)

    f = filt.get_filter(NIP, POLARIZATION, threshold=0.8, width=0.1)
    Srf = Sr * f
    Svf = Sv * f

    assert Srf[idx_5hz, idx_10sec] == pytest.approx(0.0, abs=TOL)
    assert Srf[idx_2hz, idx_10sec] == pytest.approx(0.0, abs=TOL)
    assert abs(Srf[idx_1hz, idx_10sec]) == pytest.approx(0.5, abs=TOL)

    rf = istransform(Sr * f, Fs=fs)
    tf = istransform(St * f, Fs=fs)
    vf = istransform(Sv * f, Fs=fs)

    assert tf.max() == pytest.approx(0.0, abs=TOL)
    assert rf.max() == pytest.approx(0.5, abs=TOL)
    assert vf.max() == pytest.approx(1.0, abs=TOL)


def test_scalar_azimuth_prograde(synthetic_data):
    TOL = 1.0 # [degrees]
    POLARIZATION = 'prograde'

    az_retro, n, e, v, fs = synthetic_data
    Sn = stransform(n, Fs=fs, return_time_freq=False)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)
    xpr = filt.xpr(az_retro)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = filt.instantaneous_azimuth(Sv, Sn, Se, POLARIZATION, xpr)
    Sr, St = filt.rotate_NE_RT(Sn, Se, theta)
    NIP = filt.NIP(Sr, Sv, polarization=POLARIZATION, eps=0.04)

    f = filt.get_filter(NIP, POLARIZATION, threshold=0.8, width=0.1)
    Svhat = filt.shift_phase(Sv, POLARIZATION)

    ef = istransform(Se * f, Fs=fs)
    nf = istransform(Sn * f, Fs=fs)
    vfhat = istransform(Svhat * f, Fs=fs)

    az = filt.scalar_azimuth(ef, nf, vfhat)

    assert az == pytest.approx(az_retro + 90, abs=TOL)