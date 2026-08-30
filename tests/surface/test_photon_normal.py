##########################################################################################
# tests/surface/test_photon_normal.py
##########################################################################################

import numpy as np
import pytest

from polymath               import Scalar, Vector3
from oops.event             import Event
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.path.spicepath    import SpicePath
from oops.surface.ellipsoid import Ellipsoid

C = 299792.458          # km/s, matching oops.constants.C

REQ = 6378.
RPOL = 6357.


def _angle_to(surface, cept, target):
    """Angle in radians between the surface normal at `cept` and the direction to
    `target`.

    This is the defining property of every solver in this module: the photon path is
    parallel to the surface normal at the intercept point.

    Parameters:
        surface (Surface): The Surface on which `cept` falls.
        cept (Vector3): The surface intercept point, relative to the Surface's origin and
            frame.
        target (Vector3): The remote position, relative to the same origin and frame.

    Returns:
        float: The angle in radians.
    """

    sep = surface.normal(cept).sep(target - cept)
    return abs(float(sep.vals if sep.shape == () else sep.vals[0]))


@pytest.fixture
def planet():
    """An oblate Ellipsoid centered on the SSB, so its origin does not move.

    A static origin lets a test compare against a geometric reference without having to
    correct for the origin's own motion during the light travel time.
    """

    return Ellipsoid(Path.SSB, Frame.J2000, (REQ, REQ, RPOL))


@pytest.fixture
def observer():
    """An observer position well outside the planet, relative to the SSB in J2000."""

    return Vector3([[1.e6, 0., 1.e5]])


##########################################################################################
# Solvers based on the surface normal and a remote event
##########################################################################################

def test_photon_normal_to_event_lands_on_the_surface(planet, observer):
    """The sub-observer intercept falls between the polar and equatorial radii."""

    arrival = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, _) = planet.photon_normal_to_event(arrival)

    radius = float(surface_event.pos.norm().vals[0])
    assert radius == pytest.approx(6377.794, abs=1.e-3)
    assert RPOL <= radius <= REQ


def test_photon_normal_to_event_normal_points_at_the_observer(planet, observer):
    """The surface normal at the intercept is parallel to the line to the observer."""

    arrival = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, _) = planet.photon_normal_to_event(arrival)

    assert _angle_to(planet, surface_event.pos, observer) < 1.e-12


def test_photon_normal_to_event_light_time_matches_the_distance(planet, observer):
    """`dep_lt` equals the intercept-to-observer distance divided by the speed of
    light.
    """

    arrival = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, _) = planet.photon_normal_to_event(arrival)

    distance = float((observer - surface_event.pos).norm().vals[0])
    assert float(surface_event.dep_lt.vals[0]) == pytest.approx(distance / C, rel=1.e-12)


def test_photon_normal_to_event_light_time_signs(planet, observer):
    """A photon departing the surface has `dep_lt` > 0 and `arr_lt` < 0."""

    arrival = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, arrival_event) = planet.photon_normal_to_event(arrival)

    assert float(surface_event.dep_lt.vals[0]) > 0.
    assert float(arrival_event.arr_lt.vals[0]) < 0.


def test_photon_event_to_normal_normal_points_at_the_departure(planet, observer):
    """The reversed solver puts the normal along the line to the departure event."""

    departure = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, _) = planet.photon_event_to_normal(departure)

    assert _angle_to(planet, surface_event.pos, observer) < 1.e-12


def test_photon_event_to_normal_light_time_signs(planet, observer):
    """A photon arriving at the surface has `arr_lt` < 0 and `dep_lt` > 0."""

    departure = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, departure_event) = planet.photon_event_to_normal(departure)

    assert float(surface_event.arr_lt.vals[0]) < 0.
    assert float(departure_event.dep_lt.vals[0]) > 0.


def test_photon_event_to_normal_finds_the_same_point(planet, observer):
    """Both event solvers converge on the same surface point.

    The geometry is symmetric, so reversing the photon direction cannot move the
    intercept.
    """

    event = Event(0., observer, Path.SSB, Frame.J2000)
    (to_event, _) = planet.photon_normal_to_event(event)
    (from_event, _) = planet.photon_event_to_normal(event)

    assert np.allclose(to_event.pos.vals, from_event.pos.vals, rtol=1.e-12)


def test_photon_normal_to_event_carries_surface_subfields(planet, observer):
    """The surface event carries the coordinate and normal subfields."""

    arrival = Event(0., observer, Path.SSB, Frame.J2000)
    (surface_event, _) = planet.photon_normal_to_event(arrival)

    for key in ('coord1', 'coord2', 'coord3'):
        assert key in surface_event.subfields

    assert surface_event.perp is not None
    assert surface_event.vflat is not None


##########################################################################################
# Solvers based on the surface normal and a remote path
##########################################################################################

@pytest.fixture
def sun_and_planet(core_kernels):
    """The Sun as a SpicePath, with an Earth-sized Ellipsoid centered on the SSB."""

    sun = SpicePath('SUN', 'SSB')
    return (sun, Ellipsoid(Path.SSB, Frame.J2000, (REQ, REQ, RPOL)))


def _sun_at(sun, time):
    """The Sun's position relative to the SSB in J2000 at the given time."""

    return sun.wrt(Path.SSB, Frame.J2000).event_at_time(time).pos


def test_photon_path_to_normal_lands_on_the_surface(sun_and_planet):
    """The sub-solar intercept falls between the polar and equatorial radii."""

    (sun, planet) = sun_and_planet
    (surface_event, _) = planet.photon_path_to_normal(Scalar([0.]), sun)

    radius = float(surface_event.pos.norm().vals[0])
    assert RPOL <= radius <= REQ


def test_photon_path_to_normal_normal_points_at_the_departure_position(sun_and_planet):
    """The normal points at where the Sun was when the photon departed.

    The reference is the Sun's position at the path event time, not at the surface event
    time; the two differ by the light travel time.
    """

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_path_to_normal(Scalar([0.]), sun)

    target = _sun_at(sun, path_event.time)
    assert _angle_to(planet, surface_event.pos, target) < 1.e-12


def test_photon_path_to_normal_light_time_matches_the_distance(sun_and_planet):
    """`arr_lt` equals minus the intercept-to-Sun distance divided by the speed of
    light.
    """

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_path_to_normal(Scalar([0.]), sun)

    target = _sun_at(sun, path_event.time)
    distance = float((target - surface_event.pos).norm().vals[0])
    assert float(surface_event.arr_lt.vals[0]) == pytest.approx(-distance / C, rel=1.e-9)


def test_photon_path_to_normal_light_time_signs(sun_and_planet):
    """A photon arriving at the surface has `arr_lt` < 0 and `dep_lt` > 0."""

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_path_to_normal(Scalar([0.]), sun)

    assert float(surface_event.arr_lt.vals[0]) < 0.
    assert float(path_event.dep_lt.vals[0]) > 0.


def test_photon_normal_to_path_light_time_signs(sun_and_planet):
    """A photon departing the surface has `dep_lt` > 0 and `arr_lt` < 0."""

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_normal_to_path(Scalar([0.]), sun)

    assert float(surface_event.dep_lt.vals[0]) > 0.
    assert float(path_event.arr_lt.vals[0]) < 0.


def test_photon_normal_to_path_normal_points_at_the_arrival_position(sun_and_planet):
    """The normal points at where the Sun is when the photon arrives."""

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_normal_to_path(Scalar([0.]), sun)

    target = _sun_at(sun, path_event.time)
    assert _angle_to(planet, surface_event.pos, target) < 1.e-12


def test_photon_path_to_normal_puts_the_path_event_earlier(sun_and_planet):
    """A photon arriving at the surface departed the path before it got there."""

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_path_to_normal(Scalar([0.]), sun)

    assert float(path_event.time.vals[0]) < float(surface_event.time.vals[0])


def test_photon_normal_to_path_puts_the_path_event_later(sun_and_planet):
    """A photon departing the surface reaches the path afterward."""

    (sun, planet) = sun_and_planet
    (surface_event, path_event) = planet.photon_normal_to_path(Scalar([0.]), sun)

    assert float(path_event.time.vals[0]) > float(surface_event.time.vals[0])


def test_photon_path_solvers_straddle_the_surface_time(sun_and_planet):
    """The two path solvers place their path events symmetrically about the surface
    time.

    They see the Sun at different times, so their intercepts differ slightly; what must
    match is the light travel time in each direction.
    """

    (sun, planet) = sun_and_planet
    (_, incoming) = planet.photon_path_to_normal(Scalar([0.]), sun)
    (_, outgoing) = planet.photon_normal_to_path(Scalar([0.]), sun)

    assert float(incoming.time.vals[0]) == pytest.approx(-float(outgoing.time.vals[0]),
                                                         rel=1.e-6)


def test_photon_path_to_normal_carries_surface_subfields(sun_and_planet):
    """The surface event carries the coordinate and normal subfields."""

    (sun, planet) = sun_and_planet
    (surface_event, _) = planet.photon_path_to_normal(Scalar([0.]), sun)

    for key in ('coord1', 'coord2', 'coord3'):
        assert key in surface_event.subfields

    assert surface_event.perp is not None
    assert surface_event.vflat is not None


##########################################################################################
# Exclusion zone near the center of the surface
##########################################################################################

@pytest.mark.parametrize('pos, expect_masked', [
    (Vector3((7.e4, 2.e4, 3.e4)),  False),      # shapeless, outside
    (Vector3([[7.e4, 2.e4, 3.e4]]), False),     # shaped, outside
    (Vector3((1., 1., 1.)),         True),      # shapeless, inside the exclusion zone
    (Vector3([[1., 1., 1.]]),       True),      # shaped, inside the exclusion zone
], ids=['shapeless-outside', 'shaped-outside', 'shapeless-inside', 'shaped-inside'])
def test_intercept_normal_to_handles_the_exclusion_zone(pos, expect_masked):
    """`intercept_normal_to` accepts shapeless and shaped positions alike.

    Positions inside the exclusion zone near the center are masked rather than raising.
    """

    surface = Ellipsoid(Path.SSB, Frame.J2000, (60000., 50000., 40000.))
    cept = surface.intercept_normal_to(pos)

    assert bool(np.any(cept.mask)) is expect_masked


def test_intercept_normal_to_masks_only_the_excluded_elements():
    """In a mixed array, only the elements inside the exclusion zone are masked."""

    surface = Ellipsoid(Path.SSB, Frame.J2000, (60000., 50000., 40000.))
    cept = surface.intercept_normal_to(Vector3([[7.e4, 2.e4, 3.e4], [1., 1., 1.]]))

    assert list(np.asarray(cept.mask).ravel()) == [False, True]

##########################################################################################
