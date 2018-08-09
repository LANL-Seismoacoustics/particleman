from particleman import stransform
import particleman.filter as filt

def test_xpr():
    assert filt.xpr(45) == 1
    assert filt.xpr(160) == 1
    assert filt.xpr(220) == -1
    assert filt.xpr(259) == -1

def test_instantaneous_azimuth(synthetic_data):
    n, e, v, fs, az_retro = synthetic_data
    Sn, T, F = stransform(n, Fs=fs, return_time_freq=True)
    Se = stransform(e, Fs=fs)
    Sv = stransform(v, Fs=fs)

    xpr = filt.xpr(az_retro)
    theta = filt.instantaneous_azimuth(Sv, Sn, Se, 'retrograde', xpr)


