##########################################################################################
# tests/path/test_linearcoordpath.py
##########################################################################################

import pytest

import numpy as np

from polymath     import Vector3
from oops.path    import LinearCoordPath, FixedPath, Path
from oops.surface import Ansa, RingPlane

RADIUS = 1.e5
LONGITUDE = 0.3
DLON_DT = 1.e-6
EPOCH = 0.
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


@pytest.fixture
def path(ringplane):
    """A ring-plane path at a fixed radius, drifting in longitude."""

    return LinearCoordPath(ringplane, (RADIUS, LONGITUDE), (0., DLON_DT), EPOCH)


def test_linearcoordpath_matches_the_surface_at_the_epoch(path, ringplane):
    expected = ringplane.vector3_from_coords((RADIUS, LONGITUDE))

    assert path.event_at_time(EPOCH).pos.vals == pytest.approx(expected.vals, abs=1.e-9)


def test_linearcoordpath_advances_the_coordinates_linearly(path):
    dt = 1000.
    lon = LONGITUDE + DLON_DT * dt
    expected = (RADIUS * np.cos(lon), RADIUS * np.sin(lon), 0.)

    assert path.event_at_time(EPOCH + dt).pos.vals == pytest.approx(expected, abs=1.e-6)


def test_linearcoordpath_velocity_follows_the_coordinate_rate(path):
    dt = 1000.
    lon = LONGITUDE + DLON_DT * dt
    expected = (-RADIUS * DLON_DT * np.sin(lon), RADIUS * DLON_DT * np.cos(lon), 0.)

    assert path.event_at_time(EPOCH + dt).vel.vals == pytest.approx(expected, abs=1.e-12)


def test_linearcoordpath_adopts_the_surface_origin_and_frame(path, ringplane):
    assert path.origin is ringplane.origin
    assert path.frame is ringplane.frame


def test_linearcoordpath_shape_follows_the_coordinates(ringplane):
    path = LinearCoordPath(ringplane, ([RADIUS, 2. * RADIUS], LONGITUDE),
                           (0., DLON_DT), EPOCH)

    assert path.shape == (2,)
    assert path.event_at_time(EPOCH).shape == (2,)


def test_linearcoordpath_shares_one_waypoint_per_definition(path, ringplane):
    same = LinearCoordPath(ringplane, (RADIUS, LONGITUDE), (0., DLON_DT), EPOCH)
    other = LinearCoordPath(ringplane, (RADIUS, LONGITUDE), (0., 2. * DLON_DT), EPOCH)

    assert same.waypoint is path.waypoint
    assert other.waypoint is not path.waypoint


def test_linearcoordpath_waypoint_distinguishes_the_observer(path, ringplane, observer):
    with_obs = LinearCoordPath(ringplane, (RADIUS, LONGITUDE), (0., DLON_DT), EPOCH,
                               obs=observer)

    assert with_obs.waypoint is not path.waypoint


def test_linearcoordpath_survives_a_state_roundtrip(ringplane, observer):
    path = LinearCoordPath(ringplane, (RADIUS, LONGITUDE), (0., DLON_DT), EPOCH,
                           obs=observer)
    state = path.__getstate__()

    copied = Path.__new__(LinearCoordPath)
    copied.__setstate__(state)

    assert copied.__getstate__() == state


def test_linearcoordpath_rejects_a_virtual_surface_without_an_observer(ansa):
    with pytest.raises(NotImplementedError,
                       match='LinearCoordPath requires an observation path'):
        LinearCoordPath(ansa, (RADIUS, 0.), (0., 0.), EPOCH)


def test_linearcoordpath_locates_the_ansa_relative_to_the_observer(ansa, observer):
    # At theta == 0 the ansa point is the one whose position is perpendicular to the
    # line of sight from the observer, at the given projected radius.
    path = LinearCoordPath(ansa, (RADIUS, 0.), (0., 0.), EPOCH, obs=observer)
    pos = path.event_at_time(EPOCH).pos
    los = pos - Vector3(OBS_POS)
    cosine = pos.dot(los) / (pos.norm() * los.norm())

    assert pos.norm().vals == pytest.approx(RADIUS, abs=1.e-6)
    assert cosine.vals == pytest.approx(0., abs=1.e-12)

##########################################################################################
