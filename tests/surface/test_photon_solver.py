##########################################################################################
# tests/surface/test_photon_solver.py: the line-of-sight and coordinate photon solvers
##########################################################################################

import numpy as np
import pytest

from polymath               import Scalar, Vector3
from oops.event             import Event
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.ellipsoid import Ellipsoid

C = 299792.458          # km/s, matching oops.constants.C

REQ = 6378.
RPOL = 6357.


@pytest.fixture
def planet() -> Ellipsoid:
    """An oblate Ellipsoid centered on the SSB, so its origin does not move."""

    return Ellipsoid(Path.SSB, Frame.J2000, (REQ, REQ, RPOL))


@pytest.fixture
def observer() -> Vector3:
    """An observer position well outside the planet, relative to the SSB in J2000."""

    return Vector3([[1.e6, 0., 1.e5]])


def _arrival(observer: Vector3) -> Event:
    """An arrival event at the observer, receiving a photon from the planet."""

    event = Event(Scalar(0.), observer, Path.SSB, Frame.J2000)
    event.arr = observer                # the photon travels outward, toward the observer

    return event


def _departure(observer: Vector3) -> Event:
    """A departure event at the observer, sending a photon toward the planet."""

    event = Event(Scalar(0.), observer, Path.SSB, Frame.J2000)
    event.dep = -observer               # the photon travels inward, toward the planet

    return event


##########################################################################################
# photon_to_event
##########################################################################################

def test_photon_to_event_lands_on_the_surface(planet: Ellipsoid,
                                              observer: Vector3) -> None:
    """The intercept satisfies the surface equation, so its elevation is zero."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer))
    (_, _, z) = planet.coords_from_vector3(surface_event.pos, axes=3)

    assert z.vals == pytest.approx(0., abs=1.e-6)


def test_photon_to_event_lies_between_the_radii(planet: Ellipsoid,
                                                observer: Vector3) -> None:
    """A point on an oblate spheroid is no further out than the equatorial radius."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer))
    radius = surface_event.pos.norm().vals[0]

    assert RPOL <= radius <= REQ


def test_photon_to_event_departs_toward_the_observer(planet: Ellipsoid,
                                                     observer: Vector3) -> None:
    """The outgoing photon points from the surface to the arrival event."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer))
    toward_observer = (observer - surface_event.pos).unit()

    assert surface_event.dep.unit().vals \
           == pytest.approx(toward_observer.vals, abs=1.e-9)


def test_photon_to_event_light_time_matches_the_distance(planet: Ellipsoid,
                                                         observer: Vector3) -> None:
    """The travel time is the separation divided by the speed of light."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer))
    distance = (observer - surface_event.pos).norm().vals[0]

    assert surface_event.dep_lt.vals[0] == pytest.approx(distance / C, rel=1.e-9)


def test_photon_to_event_light_time_signs(planet: Ellipsoid,
                                          observer: Vector3) -> None:
    """The departure time is positive and the matching arrival time is negative."""

    (surface_event, arrival_event) = planet.photon_to_event(_arrival(observer))

    assert surface_event.dep_lt.vals[0] > 0.
    assert arrival_event.arr_lt.vals[0] == pytest.approx(-surface_event.dep_lt.vals[0])


def test_photon_to_event_precedes_the_arrival(planet: Ellipsoid,
                                              observer: Vector3) -> None:
    """The photon leaves the surface before it arrives."""

    (surface_event, arrival_event) = planet.photon_to_event(_arrival(observer))

    assert surface_event.time.vals[0] < arrival_event.time.vals[0]


def test_photon_to_event_is_in_the_surface_frame(planet: Ellipsoid,
                                                 observer: Vector3) -> None:
    """The surface event is defined relative to the Surface's origin and frame."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer))

    assert surface_event.origin == planet.origin
    assert surface_event.frame == planet.frame


def test_a_line_of_sight_that_misses_is_masked(planet: Ellipsoid,
                                               observer: Vector3) -> None:
    """A direction that never meets the surface yields a masked result."""

    event = Event(Scalar(0.), observer, Path.SSB, Frame.J2000)
    event.arr = Vector3([[1., 0., 0.]])         # pointing away from the planet

    (surface_event, _) = planet.photon_to_event(event)

    assert surface_event.pos.mask


def test_an_initial_guess_reaches_the_same_solution(planet: Ellipsoid,
                                                    observer: Vector3) -> None:
    """A guess only speeds up the iteration; it does not change the answer."""

    (without, _) = planet.photon_to_event(_arrival(observer))
    (with_guess, _) = planet.photon_to_event(_arrival(observer),
                                             guess=without.time)

    assert with_guess.pos == without.pos


def test_the_solver_converges_within_a_few_iterations(planet: Ellipsoid,
                                                      observer: Vector3) -> None:
    """The documented iteration count should almost never need to exceed six."""

    (few, _) = planet.photon_to_event(_arrival(observer),
                                      converge={'max_iterations': 2})
    (many, _) = planet.photon_to_event(_arrival(observer),
                                       converge={'max_iterations': 6})

    assert few.pos.vals == pytest.approx(many.pos.vals, abs=1.e-6)


def test_an_antimask_of_all_true_changes_nothing(planet: Ellipsoid,
                                                 observer: Vector3) -> None:
    """Keeping every element gives the same solution as no filter at all."""

    (filtered, _) = planet.photon_to_event(_arrival(observer),
                                           antimask=np.array([True]))
    (plain, _) = planet.photon_to_event(_arrival(observer))

    assert filtered.pos == plain.pos


def test_an_antimask_of_all_false_masks_everything(planet: Ellipsoid,
                                                   observer: Vector3) -> None:
    """With nothing left to solve for, the result is entirely masked."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer),
                                                antimask=np.array([False]))

    assert surface_event.pos.mask


def test_time_derivatives_are_always_retained(planet: Ellipsoid,
                                              observer: Vector3) -> None:
    """Derivatives with respect to time survive even when derivs is False."""

    (surface_event, _) = planet.photon_to_event(_arrival(observer), derivs=False)

    assert 'T' in surface_event.pos.derivs


##########################################################################################
# photon_from_event
##########################################################################################

def test_photon_from_event_lands_on_the_surface(planet: Ellipsoid,
                                                observer: Vector3) -> None:
    """The intercept satisfies the surface equation, so its elevation is zero."""

    (surface_event, _) = planet.photon_from_event(_departure(observer))
    (_, _, z) = planet.coords_from_vector3(surface_event.pos, axes=3)

    assert z.vals == pytest.approx(0., abs=1.e-6)


def test_photon_from_event_arrives_from_the_observer(planet: Ellipsoid,
                                                     observer: Vector3) -> None:
    """The incoming photon points from the departure event to the surface."""

    (surface_event, _) = planet.photon_from_event(_departure(observer))
    toward_surface = (surface_event.pos - observer).unit()

    assert surface_event.arr.unit().vals == pytest.approx(toward_surface.vals, abs=1.e-9)


def test_photon_from_event_light_time_signs(planet: Ellipsoid,
                                            observer: Vector3) -> None:
    """The arrival time is negative and the matching departure time is positive."""

    (surface_event, departure_event) = planet.photon_from_event(_departure(observer))

    assert surface_event.arr_lt.vals[0] < 0.
    assert departure_event.dep_lt.vals[0] == pytest.approx(-surface_event.arr_lt.vals[0])


def test_photon_from_event_follows_the_departure(planet: Ellipsoid,
                                                 observer: Vector3) -> None:
    """The photon reaches the surface after it leaves the observer."""

    (surface_event, departure_event) = planet.photon_from_event(_departure(observer))

    assert surface_event.time.vals[0] > departure_event.time.vals[0]


def test_both_solvers_find_a_point_on_the_line_of_sight(planet: Ellipsoid,
                                                       observer: Vector3) -> None:
    """The line here passes through the planet's center, so each intercept is
    parallel to the direction from the observer to that center."""

    (outbound, _) = planet.photon_to_event(_arrival(observer))
    (inbound, _) = planet.photon_from_event(_departure(observer))

    assert outbound.pos.unit().cross(observer.unit()).norm().vals \
           == pytest.approx(0., abs=1.e-9)
    assert inbound.pos.unit().cross(observer.unit()).norm().vals \
           == pytest.approx(0., abs=1.e-9)


def test_both_solvers_find_a_point_the_same_distance_from_the_center(
        planet: Ellipsoid, observer: Vector3) -> None:
    """The line passes through the center, so the two intercepts are antipodal."""

    (outbound, _) = planet.photon_to_event(_arrival(observer))
    (inbound, _) = planet.photon_from_event(_departure(observer))

    assert inbound.pos.norm().vals == pytest.approx(outbound.pos.norm().vals, abs=1.e-3)


##########################################################################################
# photon_to_coords and photon_from_coords
##########################################################################################

def test_photon_to_coords_reaches_the_named_point(planet: Ellipsoid,
                                                  observer: Vector3) -> None:
    """The surface event falls at the coordinates that were asked for."""

    (reference, _) = planet.photon_to_event(_arrival(observer))
    coords = planet.coords_from_vector3(reference.pos, axes=2)

    (surface_event, _) = planet.photon_to_coords(_arrival(observer), coords)

    assert surface_event.pos.vals == pytest.approx(reference.pos.vals, abs=1.e-3)


def test_photon_from_coords_reaches_the_named_point(planet: Ellipsoid,
                                                    observer: Vector3) -> None:
    """The surface event falls at the coordinates that were asked for."""

    (reference, _) = planet.photon_from_event(_departure(observer))
    coords = planet.coords_from_vector3(reference.pos, axes=2)

    (surface_event, _) = planet.photon_from_coords(_departure(observer), coords)

    assert surface_event.pos.vals == pytest.approx(reference.pos.vals, abs=1.e-3)


def test_photon_to_coords_light_time_matches_the_distance(planet: Ellipsoid,
                                                          observer: Vector3) -> None:
    """The travel time is the separation divided by the speed of light."""

    (reference, _) = planet.photon_to_event(_arrival(observer))
    coords = planet.coords_from_vector3(reference.pos, axes=2)

    (surface_event, _) = planet.photon_to_coords(_arrival(observer), coords)
    distance = (observer - surface_event.pos).norm().vals[0]

    assert surface_event.dep_lt.vals[0] == pytest.approx(distance / C, rel=1.e-6)


def test_photon_to_coords_drops_the_masked_coordinates_with_the_link(
                                                            planet: Ellipsoid) -> None:
    """A partly masked link leaves the coordinates indexed alongside it.

    The solver shrinks the link event to its unmasked elements before iterating, so the
    coordinates have to be shrunk by the same antimask; the result is unshrunk back to
    the original shape. With an unmasked link the shrink is a no-op, so only a link
    carrying a masked element exercises this.
    """

    observer = Vector3([[1.e6, 0., 1.e5], [1.e6, 0., 1.e5], [1.e6, 2.e5, 1.e5]],
                       mask=[False, True, False])
    (reference, _) = planet.photon_to_event(_arrival(observer))
    coords = planet.coords_from_vector3(reference.pos, axes=2)

    (surface_event, _) = planet.photon_to_coords(_arrival(observer), coords)

    assert surface_event.shape == (3,)
    assert surface_event.pos.vals[surface_event.pos.antimask] == pytest.approx(
                            reference.pos.vals[reference.pos.antimask], abs=1.e-3)

##########################################################################################
