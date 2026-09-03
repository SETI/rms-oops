##########################################################################################
# tests/backplane/test_ring.py
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane
from oops.constants import TWOPI

PLANET = 'SATURN'
RING = 'SATURN:RING'

# radial_mode takes a fully-specified ring_radius key, including its rmin and rmax.
RADIUS_KEY = ('ring_radius', RING, None, None)

HALFPI = np.pi / 2.

# The mid-time of the synthetic Saturn snapshot.
EPOCH = 1.e8 + 5.


def _unmasked(array) -> np.ndarray:
    """The values of a backplane where it is not masked."""

    return array.vals[array.antimask]


def test_ring_radius_is_positive(bp: Backplane) -> None:
    """A ring radius is a distance from the planet's center."""

    assert np.all(_unmasked(bp.ring_radius(RING)) >= 0.)


def test_ring_radius_spans_the_field_of_view(bp: Backplane) -> None:
    """The 40x40 field reaches well beyond the main rings."""

    radii = _unmasked(bp.ring_radius(RING))

    assert radii.min() < 60000.
    assert radii.max() > 100000.


def test_ring_radius_limits_mask_the_rest(bp: Backplane) -> None:
    """rmin and rmax restrict the backplane to an annulus."""

    limited = bp.ring_radius(RING, rmin=80000., rmax=100000.)
    values = _unmasked(limited)

    assert np.all(values >= 80000.)
    assert np.all(values <= 100000.)
    assert np.any(limited.mask)


def test_ring_longitude_spans_the_full_circle(bp: Backplane) -> None:
    """Longitudes are angles, reported between 0 and 2*pi."""

    longitudes = _unmasked(bp.ring_longitude(RING))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


@pytest.mark.parametrize('reference', ['aries', 'node', 'obs', 'sun', 'oha', 'sha'])
def test_ring_longitude_accepts_every_reference(reference: str, bp: Backplane) -> None:
    """Each reference shifts the zero point of longitude."""

    longitudes = _unmasked(bp.ring_longitude(RING, reference=reference))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


def test_opposite_references_are_half_a_turn_apart(bp: Backplane) -> None:
    """'oha' is the anti-observer longitude, half a turn from 'obs'."""

    obs = _unmasked(bp.ring_longitude(RING, reference='obs'))
    oha = _unmasked(bp.ring_longitude(RING, reference='oha'))
    difference = (obs - oha) % TWOPI

    assert np.allclose(difference, np.pi)


def test_ring_elevation_is_the_complement_of_the_emission_angle(bp: Backplane) -> None:
    """The elevation to the observer is pi/2 minus the emission angle."""

    elevation = _unmasked(bp.ring_elevation(RING, direction='obs', pole='prograde'))
    emission = _unmasked(bp.ring_emission_angle(RING, pole='prograde'))

    assert np.allclose(elevation, HALFPI - emission)


def test_ring_elevation_toward_the_sun_complements_the_incidence_angle(
        bp: Backplane) -> None:
    """The elevation to the Sun is pi/2 minus the incidence angle."""

    elevation = _unmasked(bp.ring_elevation(RING, direction='sun', pole='prograde'))
    incidence = _unmasked(bp.ring_incidence_angle(RING, pole='prograde'))

    assert np.allclose(elevation, HALFPI - incidence)


def test_ring_incidence_angle_is_sunward_by_default(bp: Backplane) -> None:
    """Measured from the sunward pole, the incidence angle is always at most pi/2."""

    assert np.all(_unmasked(bp.ring_incidence_angle(RING)) <= HALFPI + 1.e-12)


@pytest.mark.parametrize('pole', ['sunward', 'observed', 'north', 'prograde'])
def test_ring_incidence_angle_accepts_every_pole(pole: str, bp: Backplane) -> None:
    """Each choice of pole gives an angle between 0 and pi."""

    values = _unmasked(bp.ring_incidence_angle(RING, pole=pole))

    assert np.all(values >= 0.)
    assert np.all(values <= np.pi)


def test_the_north_and_prograde_incidence_angles_are_supplementary_or_equal(
        bp: Backplane) -> None:
    """Saturn's rings are prograde about its IAU north pole, so the two agree."""

    north = _unmasked(bp.ring_incidence_angle(RING, pole='north'))
    prograde = _unmasked(bp.ring_incidence_angle(RING, pole='prograde'))

    assert np.allclose(north, prograde) or np.allclose(north, np.pi - prograde)


def test_ring_azimuth_spans_the_full_circle(bp: Backplane) -> None:
    """The azimuth is measured in the prograde direction, so it wraps at 2*pi."""

    azimuth = _unmasked(bp.ring_azimuth(RING))

    assert np.all(azimuth >= 0.)
    assert np.all(azimuth < TWOPI)


@pytest.mark.parametrize('direction', ['obs', 'sun'])
def test_ring_azimuth_accepts_both_directions(direction: str, bp: Backplane) -> None:
    """The azimuth can be measured from the observer's or the Sun's direction."""

    azimuth = _unmasked(bp.ring_azimuth(RING, direction=direction))

    assert np.all(azimuth >= 0.)
    assert np.all(azimuth < TWOPI)


def test_ring_radial_resolution_is_positive(bp: Backplane) -> None:
    """Resolution is a length per pixel, so it is never negative."""

    assert np.all(_unmasked(bp.ring_radial_resolution(RING)) > 0.)


def test_ring_angular_resolution_in_km_scales_with_radius(bp: Backplane) -> None:
    """Converting radians per pixel to km per pixel multiplies by the radius."""

    radians = bp.ring_angular_resolution(RING, units='rad')
    km = bp.ring_angular_resolution(RING, units='km')
    radius = bp.ring_radius(RING)
    antimask = km.antimask

    assert np.allclose(km.vals[antimask],
                       radians.vals[antimask] * radius.vals[antimask])


def test_ring_gradient_angle_is_an_angle(bp: Backplane) -> None:
    """The gradient direction is measured from the U-axis toward the V-axis."""

    angles = _unmasked(bp.ring_gradient_angle(RING))

    assert np.all(np.abs(angles) <= TWOPI)


def test_radial_mode_of_zero_amplitude_leaves_the_radius_alone(bp: Backplane) -> None:
    """A mode with no amplitude shifts nothing."""

    plain = bp.ring_radius(RING)
    moded = bp.radial_mode(RADIUS_KEY, 2, EPOCH, 0., 0., 0.)

    assert np.allclose(_unmasked(moded), _unmasked(plain))


def test_radial_mode_shifts_the_radius_by_at_most_its_amplitude(bp: Backplane) -> None:
    """A mode of amplitude A moves each radius by no more than A."""

    amplitude = 100.
    plain = bp.ring_radius(RING)
    moded = bp.radial_mode(RADIUS_KEY, 2, EPOCH, amplitude, 0., 0.)
    antimask = moded.antimask

    shift = np.abs(moded.vals[antimask] - plain.vals[antimask])
    assert shift.max() <= amplitude + 1.e-9
    assert shift.max() > 0.


def test_radial_mode_with_zero_cycles_is_a_uniform_shift(bp: Backplane) -> None:
    """With cycles == 0, every particle is at pericenter at phase zero."""

    amplitude = 100.
    plain = bp.ring_radius(RING)
    moded = bp.radial_mode(RADIUS_KEY, 0, EPOCH, amplitude, 0., 0.)
    antimask = moded.antimask

    shift = moded.vals[antimask] - plain.vals[antimask]
    assert np.allclose(shift, shift[0])


@pytest.mark.parametrize('method', ['ring_sub_observer_longitude',
                                    'ring_sub_solar_longitude',
                                    'ring_center_incidence_angle',
                                    'ring_center_emission_angle'])
def test_ring_center_backplanes_are_gridless(method: str, bp: Backplane) -> None:
    """A center backplane refers to the ring system's path, with no spatial extent."""

    assert getattr(bp, method)(RING).shape == ()


def test_ring_sub_observer_longitude_is_zero_in_its_own_reference(
        bp: Backplane) -> None:
    """Measured from the sub-observer longitude, the sub-observer point is at zero."""

    value = bp.ring_sub_observer_longitude(RING, reference='obs')

    assert value.vals == pytest.approx(0., abs=1.e-9)


def test_ring_sub_solar_longitude_is_zero_in_its_own_reference(bp: Backplane) -> None:
    """Measured from the sub-solar longitude, the sub-solar point is at zero."""

    value = bp.ring_sub_solar_longitude(RING, reference='sun')

    assert value.vals == pytest.approx(0., abs=1.e-9)


def test_ring_center_emission_angle_matches_the_ring(bp: Backplane) -> None:
    """The emission angle at the center falls within the range across the rings."""

    values = _unmasked(bp.ring_emission_angle(RING))
    center = bp.ring_center_emission_angle(RING)

    assert values.min() <= center.vals <= values.max()


def test_ring_shadow_radius_is_masked_off_the_shadow(bp: Backplane) -> None:
    """The value is masked wherever the arriving photon misses the ring plane."""

    shadow = bp.ring_shadow_radius(PLANET, RING)

    assert np.any(shadow.mask)
    assert np.all(_unmasked(shadow) >= 0.)


def test_ring_radius_in_front_is_masked_where_no_ring_intervenes(bp: Backplane) -> None:
    """The value is masked wherever the ring plane is not between body and observer."""

    in_front = bp.ring_radius_in_front(PLANET, RING)

    assert np.any(in_front.mask)
    assert np.all(_unmasked(in_front) >= 0.)


def test_ring_backplanes_are_cached(bp: Backplane) -> None:
    """A backplane already computed is returned rather than recomputed."""

    assert bp.ring_radius(RING) is bp.ring_radius(RING)
    assert bp.ring_longitude(RING) is bp.ring_longitude(RING)

##########################################################################################
