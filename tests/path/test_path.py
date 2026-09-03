##########################################################################################
# tests/path/test_path.py
##########################################################################################

import numpy as np
import pickle
import pytest

import cspyce

from polymath    import Scalar, Vector3
from oops.config import QUICK
from oops.event  import Event
from oops.frame  import Frame, SpiceFrame
from oops.path   import (Path, LinkedPath, NullPath, ReversedPath, RelativePath,
                         RotatedPath, QuickPath, LinearPath, SpicePath)
from programs.gold_master.test_support import TEST_SPICE_PREFIX


@pytest.fixture(autouse=True)
def _ephemeris_kernel():
    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('de421.bsp'))
    Path._reset_caches()
    Frame._reset_caches()

def test_path():
    Path._USE_QUICKPATHS = False

    assert Path._PATH_REGISTRY['SSB'] == Path.SSB

    # LinkedPath tests
    _ = SpicePath('SUN', 'SSB')
    earth = SpicePath('EARTH', 'SUN')

    moon = SpicePath('MOON', 'EARTH')
    linked = LinkedPath(moon, earth)

    direct = SpicePath('MOON', 'SUN')

    times = np.arange(-3.e8, 3.01e8, 0.5e7)

    direct_event = direct.event_at_time(times)
    linked_event = linked.event_at_time(times)

    eps = 1.e-6
    assert ((linked_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((linked_event.vel - direct_event.vel).norm() <= eps).all()

    # RelativePath
    relative = RelativePath(linked, SpicePath('MARS', 'SUN'))
    direct = SpicePath('MOON', 'MARS')

    direct_event = direct.event_at_time(times)
    relative_event = relative.event_at_time(times)

    eps = 1.e-6
    assert ((relative_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((relative_event.vel - direct_event.vel).norm() <= eps).all()

    # ReversedPath
    reversed = ReversedPath(relative)
    direct = SpicePath('MARS', 'MOON')

    direct_event = direct.event_at_time(times)
    reversed_event = reversed.event_at_time(times)

    eps = 1.e-6
    assert ((reversed_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((reversed_event.vel - direct_event.vel).norm() <= eps).all()

    # RotatedPath
    rotated = RotatedPath(reversed, SpiceFrame('B1950'))
    direct = SpicePath('MARS', 'MOON', 'B1950')

    direct_event = direct.event_at_time(times)
    rotated_event = rotated.event_at_time(times)

    eps = 1.e-6
    assert ((rotated_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((rotated_event.vel - direct_event.vel).norm() <= eps).all()

    # QuickPath tests
    moon = SpicePath('MOON', 'EARTH')
    quick = QuickPath(moon, -5., 5., QUICK.dictionary)

    # Perfect precision is impossible
    try:
        quick = QuickPath(moon, 0., 100.,
                          dict(QUICK.dictionary, **{'path_self_check':0.}))
        assert False, 'No ValueError raised for PRECISION = 0.'
    except ValueError:
        pass

    # Timing tests...
    test = np.zeros(3000000)
    # _ = moon.event_at_time(test, quick=False)       # takes about 15 sec
    _ = quick.event_at_time(test)                   # takes maybe 2 sec

    Path._reset_caches()
    Frame._reset_caches()

    ######################################################################################
    # Test unregistered paths
    ######################################################################################

    ssb = Path.as_waypoint('SSB')

    slider1 = LinearPath(([3,0,0],[0,3,0]), 0., ssb)
    assert not slider1.is_registered

    event = slider1.event_at_time(1.)
    assert event.pos == (3,3,0)
    assert event.vel == (0,3,0)

    slider2 = LinearPath(([-2,0,0],[0,0,-2]), 0., slider1)
    assert not slider2.is_registered

    event = slider2.event_at_time(1.)
    assert event.pos == (-2,0,-2)
    assert event.vel == (0,0,-2)

    slider3 = LinearPath(([-1,0,0],[0,-3,2]), 0., slider2)
    assert not slider3.is_registered

    event = slider3.event_at_time(1.)
    assert event.pos == (-1,-3,2)
    assert event.vel == ( 0,-3,2)

    # Link unregistered frame to registered frame
    static = slider3.wrt(ssb)

    event = static.event_at_time(1.)
    assert event.pos == (0,0,0)
    assert event.vel == (0,0,0)

    # Link registered frame to unregistered frame
    static = ssb.wrt(slider3)

    event = static.event_at_time(1.)
    assert event.pos == (0,0,0)
    assert event.vel == (0,0,0)


def test_ssb_path_is_a_pickled_singleton():
    restored = pickle.loads(pickle.dumps(Path.SSB))

    assert restored is Path.SSB
    assert restored.path_id == 'SSB'

##########################################################################################
# The internal Path classes that link one path to another
##########################################################################################

# A time at which the DE421 ephemeris is defined
TIME = Scalar([0., 1000.])


def _linked_paths() -> dict:
    """One instance of each internal linking Path, keyed by class name.

    Returns:
        dict: The paths to test.
    """

    sun = SpicePath('SUN', 'SSB')
    earth = SpicePath('EARTH', 'SSB')
    mars = SpicePath('MARS', 'EARTH')

    return {
        'NullPath':     NullPath(sun),
        'LinkedPath':   LinkedPath(mars, earth),
        'RelativePath': RelativePath(sun, earth),
        'ReversedPath': ReversedPath(mars),
        'RotatedPath':  RotatedPath(sun, SpiceFrame('IAU_EARTH')),
    }


PATH_NAMES = ['NullPath', 'LinkedPath', 'RelativePath', 'ReversedPath', 'RotatedPath']


@pytest.mark.parametrize('name', PATH_NAMES)
def test_a_linking_path_survives_a_round_trip_through_pickle(name: str) -> None:
    """Unpickling rebuilds the path and reproduces the event it defines."""

    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('pck00010.tpc'))
    path = _linked_paths()[name]
    expected = path.event_at_time(TIME).pos

    restored = pickle.loads(pickle.dumps(path))

    assert type(restored) is type(path)
    assert restored.event_at_time(TIME).pos == expected


def test_a_null_path_names_the_path_it_stands_in_for() -> None:
    """A NullPath prints the ID of the path whose position it reports."""

    path = NullPath(SpicePath('SUN', 'SSB'))

    assert str(path) == 'NullPath(SUN)'


def test_a_linked_path_requires_the_origin_of_the_first_to_be_the_second() -> None:
    """The two paths have to meet, or there is nothing to link."""

    sun = SpicePath('SUN', 'SSB')
    mars = SpicePath('MARS', 'SSB')

    with pytest.raises(ValueError, match='LinkedPath path mismatch'):
        LinkedPath(sun, mars)


def test_a_relative_path_requires_a_common_origin() -> None:
    """Two paths can only be differenced if they share an origin."""

    sun = SpicePath('SUN', 'SSB')
    SpicePath('EARTH', 'SSB')
    mars = SpicePath('MARS', 'EARTH')

    with pytest.raises(ValueError, match='RelativePath origin mismatch'):
        RelativePath(sun, mars)


def test_a_relative_path_rotates_into_the_frame_of_the_new_origin() -> None:
    """When the two paths use different frames, the result is rotated into one."""

    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('pck00010.tpc'))
    sun = SpicePath('SUN', 'SSB')
    rotated = SpicePath('EARTH', 'SSB', SpiceFrame('IAU_EARTH'))

    path = RelativePath(sun, rotated)

    assert path._rotation is not None
    assert path.event_at_time(TIME).frame is rotated.frame


def test_subtracting_an_event_needs_a_common_origin() -> None:
    """The event has to be measured from the same origin as this path."""

    sun = SpicePath('SUN', 'SSB')
    event = Event(Scalar(0.), Vector3((1., 0., 0.)), SpicePath('EARTH', 'SSB'), 'J2000')

    with pytest.raises(ValueError, match='common origin for path subtraction'):
        sun.subtract_from_event(event)


def test_subtracting_an_event_needs_a_common_frame() -> None:
    """The event has to be expressed in the same frame as this path."""

    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('pck00010.tpc'))
    sun = SpicePath('SUN', 'SSB')
    event = Event(Scalar(0.), Vector3((1., 0., 0.)), 'SSB', SpiceFrame('IAU_EARTH'))

    with pytest.raises(ValueError, match='common frame for path subtraction'):
        sun.subtract_from_event(event)


def test_subtracting_an_event_can_drop_the_derivatives() -> None:
    """Without derivs, both events are stripped before they are differenced."""

    sun = SpicePath('SUN', 'SSB')
    event = Event(Scalar(0.), Vector3((1., 0., 0.)), 'SSB', 'J2000')
    event.state.insert_deriv('pos', Vector3.IDENTITY)

    assert 'pos' not in sun.subtract_from_event(event, derivs=False).state.derivs
    assert 'pos' in sun.subtract_from_event(event, derivs=True).state.derivs


def test_adding_an_event_needs_it_to_start_at_this_path() -> None:
    """The event has to be measured from this path for the sum to mean anything."""

    sun = SpicePath('SUN', 'SSB')
    event = Event(Scalar(0.), Vector3((1., 0., 0.)), SpicePath('EARTH', 'SSB'), 'J2000')

    with pytest.raises(ValueError, match='origin must match this path'):
        sun.add_to_event(event)


def test_adding_an_event_needs_a_common_frame() -> None:
    """The event has to be expressed in the same frame as this path."""

    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('pck00010.tpc'))
    sun = SpicePath('SUN', 'SSB')
    event = Event(Scalar(0.), Vector3((1., 0., 0.)), sun, SpiceFrame('IAU_EARTH'))

    with pytest.raises(ValueError, match='common frame for path addition'):
        sun.add_to_event(event)


def test_a_path_in_another_frame_alone_is_a_rotated_path() -> None:
    """With the origin unchanged, only a rotation into the new frame is needed."""

    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('pck00010.tpc'))
    fixed = LinearPath((Vector3((1.e5, 0., 0.)), Vector3.ZERO), 0., 'SSB',
                       path_id='TEST_ROTATED_SELF')

    path = fixed.wrt('SSB', SpiceFrame('IAU_EARTH'))
    event = path.event_at_time(TIME)

    assert isinstance(path, RotatedPath)
    assert event.frame is SpiceFrame('IAU_EARTH').wayframe
    assert event.pos.norm().vals == pytest.approx([1.e5, 1.e5], rel=1.e-12)


def test_a_description_names_the_frame_when_it_is_not_the_origin_frame() -> None:
    """A path in a frame of its own reports that frame in brackets."""

    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('pck00010.tpc'))
    path = SpicePath('SUN', 'EARTH', SpiceFrame('IAU_EARTH'), path_id='TEST_ROTATED_SUN')

    assert str(path) == 'SpicePath([TEST_ROTATED_SUN-EARTH]/IAU_EARTH)'

##########################################################################################
