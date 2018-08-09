import particleman.filter as filt

def test_xpr():
    assert filt.xpr(45) == 1
    assert filt.xpr(160) == 1
    assert filt.xpr(220) == -1
    assert filt.xpr(259) == -1