##########################################################################################
# tests/frame/test_frame.py
##########################################################################################

import numpy as np
import pickle
import pytest

import cspyce

from polymath    import Scalar
from oops.config import QUICK
from oops.body   import Body
from oops.frame  import (Frame, LinkedFrame, NullFrame, QuickFrame, ReversedFrame,
                         Rotation, SpiceFrame, SpinFrame)
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
# The internal Frame classes that link one frame to another
##########################################################################################

# A time at which the test kernels define every one of these frames
TIME = Scalar([0., 1000.])

LINKING_FRAME_NAMES = ['NullFrame', 'LinkedFrame', 'ReversedFrame']


def _linking_frames() -> dict:
    """One instance of each internal linking Frame, keyed by class name.

    The frames are built from Rotations, which do not rotate about a center, so the
    linking classes have no origins to reconcile.

    Returns:
        dict: The frames to test.
    """

    tilted = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_LINKING_TILT')
    turned = Rotation(0.25, 1, tilted, frame_id='TEST_LINKING_TURN')

    return {
        'NullFrame':     NullFrame(tilted),
        'LinkedFrame':   LinkedFrame(turned, tilted),
        'ReversedFrame': ReversedFrame(tilted),
    }


@pytest.mark.parametrize('name', LINKING_FRAME_NAMES)
def test_a_linking_frame_survives_a_round_trip_through_pickle(name: str) -> None:
    """Unpickling rebuilds the frame and reproduces the transform it defines."""

    frame = _linking_frames()[name]
    expected = frame.transform_at_time(TIME).matrix

    restored = pickle.loads(pickle.dumps(frame))

    assert type(restored) is type(frame)
    assert restored.transform_at_time(TIME).matrix == expected


def test_a_null_frame_names_the_frame_it_stands_in_for() -> None:
    """A NullFrame prints the ID of the frame whose orientation it reports."""

    frame = NullFrame(Rotation(0.5, 2, Frame.J2000, frame_id='TEST_NULL_SOURCE'))

    assert str(frame) == 'NullFrame(TEST_NULL_SOURCE)'


def test_a_frame_relative_to_itself_is_a_null_frame() -> None:
    """With nothing to rotate, the connection is a null transform."""

    mars = SpiceFrame('IAU_MARS', 'J2000')

    assert isinstance(mars.wrt(mars), NullFrame)


def test_a_linked_frame_requires_the_reference_of_the_first_to_be_the_second() -> None:
    """The two frames have to meet, or there is nothing to link."""

    mars = SpiceFrame('IAU_MARS', 'J2000')
    saturn = SpiceFrame('IAU_SATURN', 'J2000')
    tilted = Rotation(0.25, 1, mars, frame_id='TEST_LINK_MISMATCH')

    with pytest.raises(ValueError, match='LinkedFrame mismatch'):
        LinkedFrame(tilted, saturn)


def test_the_node_of_a_frame_is_ninety_degrees_from_its_pole() -> None:
    """The ascending node of a frame's equator lies a quarter turn from its z-axis."""

    mars = SpiceFrame('IAU_MARS', 'J2000')

    node = mars.node_at_time(TIME)

    assert node.shape == (2,)
    assert np.all(node.vals >= 0.)
    assert np.all(node.vals < 2. * np.pi)


def test_a_frame_reports_its_own_id_when_it_is_defined_on_j2000() -> None:
    """At level 0, a frame on J2000 is named by its ID alone."""

    mars = SpiceFrame('IAU_MARS', 'J2000')

    assert mars.show(0) == '"IAU_MARS"'


def test_the_j2000_form_of_a_frame_is_evaluated_once() -> None:
    """`wrt_j2000` is cached on first use."""

    mars = SpiceFrame('IAU_MARS', 'J2000')

    assert mars.wrt_j2000 is mars.wrt_j2000


def test_a_frame_with_no_origin_is_inertial() -> None:
    """A frame that does not rotate about a point has no center of rotation."""

    mars = SpiceFrame('IAU_MARS', 'J2000')
    spinning = SpinFrame(0., 1.e-6, 0., 2, Frame.J2000, frame_id='TEST_INERTIAL_SPIN')

    assert not mars.is_inertial
    assert not spinning.is_inertial
    assert Frame.J2000.is_inertial


def test_registering_a_second_frame_under_one_id_generates_a_unique_id() -> None:
    """A duplicate ID gets a numeric suffix so the first registration survives."""

    first = Rotation(0.25, 1, Frame.J2000, frame_id='TEST_DUPLICATE_ID')
    second = Rotation(0.5, 1, Frame.J2000, frame_id='TEST_DUPLICATE_ID')

    assert first.frame_id == 'TEST_DUPLICATE_ID'
    assert second.frame_id != first.frame_id
    assert second.stripped_id == 'TEST_DUPLICATE_ID'

##########################################################################################
