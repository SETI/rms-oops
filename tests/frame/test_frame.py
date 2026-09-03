##########################################################################################
# tests/frame/test_frame.py
##########################################################################################

import numpy as np
import pickle
import pytest

import cspyce

from oops.config import QUICK
from oops.body   import Body
from oops.frame  import Frame, QuickFrame, Rotation, SpiceFrame
from oops.path   import SpicePath
from programs.gold_master.test_support import TEST_SPICE_PREFIX


@pytest.fixture(autouse=True)
def _frames_with_kernels():
    Body._undefine_solar_system()
    paths = TEST_SPICE_PREFIX.retrieve(['naif0009.tls',
                                        'pck00010.tpc',
                                        'de421.bsp'])
    for path in paths:
        cspyce.furnsh(path)
    Frame._reset_caches()

def test_frame():
    # QuickFrame tests

    _ = SpicePath('EARTH', 'SSB')
    _ = SpicePath('MOON', 'SSB')
    _ = SpiceFrame('IAU_EARTH', 'J2000')
    moon  = SpiceFrame('IAU_MOON', 'IAU_EARTH')
    quick = QuickFrame(moon, -5., 5.,
                       dict(QUICK.dictionary, **{'frame_self_check':3.e-14}))

    # Perfect precision is impossible
    with pytest.raises(ValueError):
        _ = QuickFrame(moon, -5., 5.,
                       dict(QUICK.dictionary, **{'frame_self_check':0.}))

    # Timing tests...
    test = np.zeros(200000)
    # _ = moon.transform_at_time(test, quick=False)   # takes about 10 sec
    _ = quick.transform_at_time(test)           # takes way less than 1 sec

    Frame._reset_caches()

    ######################################################################################
    # Test unregistered frames
    ######################################################################################

    j2000 = Frame.as_wayframe('J2000')
    rot_180 = Rotation(np.pi, 2, j2000)

    xform = rot_180.transform_at_time(0.)
    assert xform.matrix.vals[0,0] == pytest.approx(-1, abs=1e-14)
    assert xform.matrix.vals[0,1] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,0] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,1] == pytest.approx(-1, abs=1e-14)
    assert xform.matrix.vals[2,0] == 0
    assert xform.matrix.vals[2,1] == 0
    assert xform.matrix.vals[0,2] == 0
    assert xform.matrix.vals[1,2] == 0
    assert xform.matrix.vals[2,2] == 1

    rot_neg60 = Rotation(-np.pi/3, 2, rot_180)
    c60 = 0.5
    s60 = np.sqrt(0.75)

    xform = rot_neg60.transform_at_time(0.)
    assert xform.matrix.vals[0,0] == pytest.approx(c60, abs=1e-14)
    assert xform.matrix.vals[0,1] == pytest.approx(-s60, abs=1e-14)
    assert xform.matrix.vals[1,0] == pytest.approx(s60, abs=1e-14)
    assert xform.matrix.vals[1,1] == pytest.approx(c60, abs=1e-14)
    assert xform.matrix.vals[2,0] == 0
    assert xform.matrix.vals[2,1] == 0
    assert xform.matrix.vals[0,2] == 0
    assert xform.matrix.vals[1,2] == 0
    assert xform.matrix.vals[2,2] == 1

    rot_neg120 = Rotation(-np.pi/1.5, 2, rot_neg60)

    xform = rot_neg120.transform_at_time(0.)
    assert xform.matrix.vals[0,0] == pytest.approx(-c60, abs=1e-14)
    assert xform.matrix.vals[0,1] == pytest.approx(-s60, abs=1e-14)
    assert xform.matrix.vals[1,0] == pytest.approx(s60, abs=1e-14)
    assert xform.matrix.vals[1,1] == pytest.approx(-c60, abs=1e-14)
    assert xform.matrix.vals[2,0] == 0
    assert xform.matrix.vals[2,1] == 0
    assert xform.matrix.vals[0,2] == 0
    assert xform.matrix.vals[1,2] == 0
    assert xform.matrix.vals[2,2] == 1

    # Attempt to register a frame defined relative to an unregistered frame
    # This is no longer an error
    # self.assertRaises(ValueError, Rotation, -np.pi, 2, rot_neg60, frame_id='NEG180')

    # Link unregistered frame to registered frame
    identity = rot_neg120.wrt('J2000')

    xform = identity.transform_at_time(0.)
    assert xform.matrix.vals[0,0] == pytest.approx(1, abs=1e-14)
    assert xform.matrix.vals[0,1] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,0] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,1] == pytest.approx(1, abs=1e-14)
    assert xform.matrix.vals[2,0] == 0
    assert xform.matrix.vals[2,1] == 0
    assert xform.matrix.vals[0,2] == 0
    assert xform.matrix.vals[1,2] == 0
    assert xform.matrix.vals[2,2] == 1

    # Link registered frame to unregistered frame
    identity = Frame.J2000.wrt(rot_neg120)

    xform = identity.transform_at_time(0.)
    assert xform.matrix.vals[0,0] == pytest.approx(1, abs=1e-14)
    assert xform.matrix.vals[0,1] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,0] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,1] == pytest.approx(1, abs=1e-14)
    assert xform.matrix.vals[2,0] == 0
    assert xform.matrix.vals[2,1] == 0
    assert xform.matrix.vals[0,2] == 0
    assert xform.matrix.vals[1,2] == 0
    assert xform.matrix.vals[2,2] == 1

    # Link unregistered frame to registered frame
    identity = rot_neg120.wrt(rot_180)

    xform = identity.transform_at_time(0.)
    assert xform.matrix.vals[0,0] == pytest.approx(-1, abs=1e-14)
    assert xform.matrix.vals[0,1] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,0] == pytest.approx(0, abs=1e-14)
    assert xform.matrix.vals[1,1] == pytest.approx(-1, abs=1e-14)
    assert xform.matrix.vals[2,0] == 0
    assert xform.matrix.vals[2,1] == 0
    assert xform.matrix.vals[0,2] == 0
    assert xform.matrix.vals[1,2] == 0
    assert xform.matrix.vals[2,2] == 1


def test_j2000_frame_is_a_pickled_singleton():
    restored = pickle.loads(pickle.dumps(Frame.J2000))

    assert restored is Frame.J2000
    assert restored.frame_id == 'J2000'

##########################################################################################
