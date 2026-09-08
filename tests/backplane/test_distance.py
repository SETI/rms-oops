##########################################################################################
# tests/backplane/test_distance.py
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane
from oops.constants import C

PLANET = 'SATURN'
RING = 'SATURN:RING'

# The observation time of the synthetic Saturn snapshot, and its exposure.
TIME = 1.e8
TEXP = 10.


def _unmasked(array) -> np.ndarray:
    """The values of a backplane where it is not masked."""

    return array.vals[array.antimask]


def test_distance_is_light_time_times_the_speed_of_light(bp: Backplane) -> None:
    """Distance and light time describe the same photon path."""

    distance = _unmasked(bp.distance(PLANET))
    light_time = _unmasked(bp.light_time(PLANET))

    assert np.allclose(distance, light_time * C)


def test_departing_distance_is_the_range_to_the_observer(bp: Backplane) -> None:
    """Saturn is roughly 1.3 billion km from Earth at this epoch."""

    distance = _unmasked(bp.distance(PLANET))

    assert np.all(distance > 1.2e9)
    assert np.all(distance < 1.5e9)


def test_arriving_distance_is_the_range_to_the_sun(bp: Backplane) -> None:
    """The arriving photon has traveled from the Sun, about 1.4 billion km away."""

    distance = _unmasked(bp.distance(PLANET, direction='arr'))

    assert np.all(distance > 1.2e9)
    assert np.all(distance < 1.7e9)


def test_the_near_side_of_the_disk_is_closer(bp: Backplane) -> None:
    """The distance varies across the disk by about the radius of the planet."""

    distance = _unmasked(bp.distance(PLANET))

    assert 0. < distance.max() - distance.min() < 2.e5


def test_light_time_is_about_an_hour(bp: Backplane) -> None:
    """Light takes over an hour to reach Earth from Saturn."""

    light_time = _unmasked(bp.light_time(PLANET))

    assert np.all(light_time > 3600.)
    assert np.all(light_time < 5400.)


def test_event_time_precedes_the_observation(bp: Backplane) -> None:
    """The photon left the surface a light travel time before it was detected."""

    event_time = _unmasked(bp.event_time(PLANET))
    light_time = _unmasked(bp.light_time(PLANET))

    assert np.allclose(event_time + light_time, TIME + TEXP / 2.)


def test_distance_has_the_shape_of_the_meshgrid(bp: Backplane) -> None:
    """A surface backplane is evaluated at every sample of the meshgrid."""

    assert bp.distance(PLANET).shape == bp._shape


def test_distance_is_masked_off_the_surface(bp: Backplane) -> None:
    """A pixel that misses the planet has no distance to report."""

    distance = bp.distance(PLANET)

    assert np.any(distance.mask)
    assert np.all(distance.antimask == bp.where_intercepted(PLANET).vals)


def test_the_ring_reaches_further_than_the_planet(bp: Backplane) -> None:
    """The ring plane extends beyond the globe in both directions."""

    ring = _unmasked(bp.distance(RING))
    planet = _unmasked(bp.distance(PLANET))

    assert ring.max() > planet.max()
    assert ring.min() < planet.min()


@pytest.mark.parametrize('method', ['center_distance', 'center_light_time',
                                    'center_time'])
def test_center_backplanes_are_gridless(method: str, bp: Backplane) -> None:
    """A center backplane refers to the body's path, so it has no spatial extent."""

    assert getattr(bp, method)(PLANET).shape == ()


def test_center_distance_falls_within_the_disk(bp: Backplane) -> None:
    """The distance to the body's center lies between the near and far limbs."""

    distance = _unmasked(bp.distance(PLANET))
    center = bp.center_distance(PLANET)

    assert distance.min() <= center.vals <= distance.max()


def test_center_distance_is_light_time_times_c(bp: Backplane) -> None:
    """The relation holds at the body's center as it does across the disk."""

    assert bp.center_distance(PLANET).vals \
           == pytest.approx(bp.center_light_time(PLANET).vals * C)


@pytest.mark.parametrize('alias, direction', [('obs', 'dep'), ('sun', 'arr')])
def test_center_distance_accepts_the_direction_aliases(alias: str, direction: str,
                                                       bp: Backplane) -> None:
    """'obs' is an alias for 'dep' and 'sun' for 'arr'."""

    assert bp.center_distance(PLANET, direction=alias) \
           == bp.center_distance(PLANET, direction=direction)


def test_center_time_precedes_the_observation(bp: Backplane) -> None:
    """The photon left the body's center a light travel time before detection."""

    assert bp.center_time(PLANET).vals + bp.center_light_time(PLANET).vals \
           == pytest.approx(TIME + TEXP / 2.)


def test_distance_backplanes_are_cached(bp: Backplane) -> None:
    """A backplane already computed is returned rather than recomputed."""

    assert bp.distance(PLANET) is bp.distance(PLANET)
    assert bp.light_time(PLANET) is bp.light_time(PLANET)

##########################################################################################
