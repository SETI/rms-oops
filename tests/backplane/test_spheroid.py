##########################################################################################
# tests/backplane/test_spheroid.py
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane
from oops.constants import TWOPI

PLANET = 'SATURN'
HALFPI = np.pi / 2.


def _unmasked(array) -> np.ndarray:
    """The values of a backplane where it is not masked."""

    return array.vals[array.antimask]


def test_longitude_spans_the_full_circle(bp: Backplane) -> None:
    """With minimum=0, longitudes run from 0 to 2*pi."""

    longitudes = _unmasked(bp.longitude(PLANET))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


def test_longitude_can_be_centered_on_zero(bp: Backplane) -> None:
    """With minimum=-180, longitudes run from -pi to pi."""

    longitudes = _unmasked(bp.longitude(PLANET, minimum=-180))

    assert np.all(longitudes >= -np.pi)
    assert np.all(longitudes < np.pi)


def test_east_and_west_longitudes_are_reflections(bp: Backplane) -> None:
    """Reversing the direction of increasing longitude negates it, modulo 2*pi."""

    west = _unmasked(bp.longitude(PLANET, direction='west'))
    east = _unmasked(bp.longitude(PLANET, direction='east'))

    assert np.allclose((west + east) % TWOPI, 0., atol=1.e-9)


@pytest.mark.parametrize('reference', ['iau', 'obs', 'sun', 'oha', 'sha'])
def test_longitude_accepts_every_reference(reference: str, bp: Backplane) -> None:
    """Each reference shifts the zero point of longitude."""

    longitudes = _unmasked(bp.longitude(PLANET, reference=reference))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


def test_opposite_references_are_half_a_turn_apart(bp: Backplane) -> None:
    """'oha' is the anti-observer longitude, half a turn from 'obs'."""

    obs = _unmasked(bp.longitude(PLANET, reference='obs'))
    oha = _unmasked(bp.longitude(PLANET, reference='oha'))

    assert np.allclose((obs - oha) % TWOPI, np.pi)


def test_longitude_rejects_an_unknown_reference(bp: Backplane) -> None:
    """The reference must be one of the documented choices."""

    with pytest.raises(ValueError):
        bp.sub_observer_longitude(PLANET, reference='nowhere')


def test_longitude_rejects_an_unknown_direction(bp: Backplane) -> None:
    """The direction must be 'east' or 'west'."""

    with pytest.raises(ValueError):
        bp.sub_observer_longitude(PLANET, direction='sideways')


def test_longitude_rejects_an_unknown_minimum(bp: Backplane) -> None:
    """The minimum must be 0 or -180 degrees."""

    with pytest.raises(ValueError):
        bp.sub_observer_longitude(PLANET, minimum=90)


def test_latitude_stays_between_the_poles(bp: Backplane) -> None:
    """Latitude runs from -pi/2 at the south pole to +pi/2 at the north."""

    latitudes = _unmasked(bp.latitude(PLANET))

    assert np.all(latitudes >= -HALFPI)
    assert np.all(latitudes <= HALFPI)


def test_graphic_latitude_exceeds_centric_in_the_north(bp: Backplane) -> None:
    """Saturn is oblate, so planetographic latitude is the larger of the two."""

    centric = bp.latitude(PLANET, lat_type='centric')
    graphic = bp.latitude(PLANET, lat_type='graphic')
    antimask = centric.antimask

    northern = antimask & (centric.vals > 0.1)
    assert np.all(graphic.vals[northern] > centric.vals[northern])


def test_centric_and_graphic_latitudes_agree_at_the_equator(bp: Backplane) -> None:
    """The two latitude types coincide where the latitude is zero."""

    centric = bp.latitude(PLANET, lat_type='centric')
    graphic = bp.latitude(PLANET, lat_type='graphic')
    antimask = centric.antimask

    near_equator = antimask & (np.abs(centric.vals) < 1.e-3)
    if np.any(near_equator):
        assert np.allclose(graphic.vals[near_equator], centric.vals[near_equator],
                           atol=1.e-2)


@pytest.mark.parametrize('method', ['sub_observer_longitude', 'sub_solar_longitude',
                                    'sub_observer_latitude', 'sub_solar_latitude'])
def test_sub_point_backplanes_are_gridless(method: str, bp: Backplane) -> None:
    """A sub-point backplane refers to the body's path, with no spatial extent."""

    assert getattr(bp, method)(PLANET).shape == ()


def test_sub_observer_longitude_is_zero_in_its_own_reference(bp: Backplane) -> None:
    """Measured from the sub-observer longitude, the sub-observer point is at zero."""

    value = bp.sub_observer_longitude(PLANET, reference='obs')

    assert value.vals == pytest.approx(0., abs=1.e-9)


def test_sub_solar_longitude_is_zero_in_its_own_reference(bp: Backplane) -> None:
    """Measured from the sub-solar longitude, the sub-solar point is at zero."""

    value = bp.sub_solar_longitude(PLANET, reference='sun')

    assert value.vals == pytest.approx(0., abs=1.e-9)


def test_sub_observer_latitude_falls_within_the_disk(bp: Backplane) -> None:
    """The sub-observer latitude lies within the range of latitudes seen."""

    latitudes = _unmasked(bp.latitude(PLANET))
    center = bp.sub_observer_latitude(PLANET)

    assert latitudes.min() <= center.vals <= latitudes.max()


def test_sub_solar_and_sub_observer_latitudes_are_close(bp: Backplane) -> None:
    """Saturn is near opposition as seen from Earth, so the two nearly coincide."""

    observer = bp.sub_observer_latitude(PLANET).vals
    solar = bp.sub_solar_latitude(PLANET).vals

    assert abs(observer - solar) < 0.05


def test_spheroid_backplanes_are_cached(bp: Backplane) -> None:
    """A backplane already computed is returned rather than recomputed."""

    assert bp.longitude(PLANET) is bp.longitude(PLANET)
    assert bp.latitude(PLANET) is bp.latitude(PLANET)

##########################################################################################
