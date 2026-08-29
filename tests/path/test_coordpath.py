##########################################################################################
# tests/path/test_coordpath.py
##########################################################################################

import pytest

from polymath       import Vector3
from oops.constants import C
from oops.event     import Event
from oops.path      import CoordPath, FixedPath, Path
from oops.surface   import Ansa, RingPlane

RADIUS = 1.e5
LONGITUDE = 0.3
OBS_POS = (0., -1.e6, 0.)


@pytest.fixture
def ringplane():
    """A solid surface centered on the SSB, requiring no SPICE kernels."""

    return RingPlane('SSB', 'J2000')


@pytest.fixture
def ansa():
    """A virtual surface, whose coordinates depend on the observer's position."""

    return Ansa('SSB', 'J2000')


@pytest.fixture
def observer():
    """An unregistered observer Path, one million km from the SSB along -Y."""

    return FixedPath(OBS_POS, 'SSB')


def test_coordpath_position_matches_the_surface(ringplane):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))
    expected = ringplane.vector3_from_coords((RADIUS, LONGITUDE))

    assert path.event_at_time(0.).pos.vals == pytest.approx(expected.vals, abs=1.e-9)


def test_coordpath_adopts_the_surface_origin_and_frame(ringplane):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))

    assert path.origin is ringplane.origin
    assert path.frame is ringplane.frame


def test_coordpath_is_stationary(ringplane):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))

    assert path.event_at_time(1.e6).pos == path.event_at_time(0.).pos
    assert path.event_at_time(1.e6).vel == Vector3.ZERO


def test_coordpath_broadcasts_over_times(ringplane):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))
    event = path.event_at_time([0., 1.e6])

    # The position is fixed, so it stays shapeless while the Event takes the shape of
    # the times.
    assert event.shape == (2,)
    assert event.time.shape == (2,)
    assert event.pos == path.event_at_time(0.).pos


def test_coordpath_shape_follows_the_coordinates(ringplane):
    path = CoordPath(ringplane, ([RADIUS, 2. * RADIUS, 3. * RADIUS], LONGITUDE))

    assert path.shape == (3,)
    assert path.event_at_time(0.).shape == (3,)


def test_coordpath_shares_one_waypoint_per_definition(ringplane):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))

    assert CoordPath(ringplane, (RADIUS, LONGITUDE)).waypoint is path.waypoint
    assert CoordPath(ringplane, (RADIUS, LONGITUDE + 0.1)).waypoint is not path.waypoint


def test_coordpath_waypoint_distinguishes_the_observer(ringplane, observer):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))

    assert CoordPath(ringplane, (RADIUS, LONGITUDE),
                     obs=observer).waypoint is not path.waypoint


def test_coordpath_registers_under_its_path_id(ringplane):
    path = CoordPath(ringplane, (2.e5, 1.5), path_id='TEST_COORDPATH')

    assert path.is_registered
    assert Path.as_path('TEST_COORDPATH') is path


def test_coordpath_survives_a_state_roundtrip(ringplane, observer):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE), obs=observer)
    state = path.__getstate__()

    copied = Path.__new__(CoordPath)
    copied.__setstate__(state)

    assert copied.__getstate__() == state


def test_coordpath_rejects_a_virtual_surface_without_an_observer(ansa):
    with pytest.raises(NotImplementedError,
                       match='CoordPath requires an observation path'):
        CoordPath(ansa, (RADIUS, 0.))


def test_coordpath_locates_the_ansa_relative_to_the_observer(ansa, observer):
    # At theta == 0 the ansa point is the one whose position is perpendicular to the
    # line of sight from the observer, at the given projected radius.
    path = CoordPath(ansa, (RADIUS, 0.), obs=observer)
    pos = path.event_at_time(0.).pos
    los = pos - Vector3(OBS_POS)
    cosine = pos.dot(los) / (pos.norm() * los.norm())

    assert pos.norm().vals == pytest.approx(RADIUS, abs=1.e-6)
    assert cosine.vals == pytest.approx(0., abs=1.e-12)


def test_coordpath_solves_for_the_photon_light_travel_time(ringplane, observer):
    path = CoordPath(ringplane, (RADIUS, LONGITUDE))
    arrival = Event(0., Vector3.ZERO, observer, 'J2000')

    (path_event, arrival_event) = path.photon_to_event(arrival)
    distance = (path.event_at_time(0.).pos - Vector3(OBS_POS)).norm()

    assert arrival_event.arr_lt.vals == pytest.approx(-distance.vals / C, abs=1.e-12)
    assert path_event.pos.vals == pytest.approx(path.event_at_time(0.).pos.vals,
                                                abs=1.e-6)

##########################################################################################
