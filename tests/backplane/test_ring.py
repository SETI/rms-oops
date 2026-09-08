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
# Argument validation, mask inheritance, caching, and the remaining reference options
##########################################################################################

LIMB = 'SATURN:LIMB'
MOON = 'MIMAS'

# A ring_radius key restricted to an annulus, so that the arrays derived from it carry a
# mask that the unrestricted ones do not
ANNULUS_KEY = ('ring_radius', RING, 80000., 100000.)

# The backplanes that accept a ring_radius key in place of an event key, inheriting its
# mask
NESTED_BACKPLANES = ['ring_longitude', 'ring_azimuth', 'ring_elevation',
                     'ring_incidence_angle', 'ring_emission_angle',
                     'ring_radial_resolution', 'ring_angular_resolution',
                     'ring_gradient_angle']


@pytest.mark.parametrize('name', NESTED_BACKPLANES)
def test_a_ring_backplane_inherits_the_mask_of_a_radius_backplane(name: str,
                                                                  bp: Backplane) -> None:
    """Given a ring_radius backplane key, the array is remasked to match it."""

    annulus = bp.ring_radius(RING, rmin=80000., rmax=100000.)

    remasked = getattr(bp, name)(ANNULUS_KEY)

    assert remasked.count_masked() == annulus.count_masked()
    assert getattr(bp, name)(RING).count_masked() < annulus.count_masked()


@pytest.mark.parametrize('name', NESTED_BACKPLANES + ['ring_radius',
                                                      'ring_sub_observer_longitude',
                                                      'ring_sub_solar_longitude',
                                                      '_aries_ring_longitude'])
def test_every_ring_backplane_is_evaluated_once_and_cached(name: str,
                                                           bp: Backplane) -> None:
    """A second request returns the array already registered, not a new one."""

    assert getattr(bp, name)(RING) is getattr(bp, name)(RING)


def test_ring_longitude_rejects_an_unknown_reference(bp: Backplane) -> None:
    """The reference must be one of the six the docstring lists."""

    with pytest.raises(ValueError, match="invalid longitude reference: 'noon'"):
        bp.ring_longitude(RING, reference='noon')


def test_ring_azimuth_rejects_an_unknown_direction(bp: Backplane) -> None:
    """The azimuth is measured toward the observer or the Sun, and nothing else."""

    with pytest.raises(ValueError, match="invalid azimuth direction: 'north'"):
        bp.ring_azimuth(RING, direction='north')


def test_ring_elevation_rejects_an_unknown_direction(bp: Backplane) -> None:
    """The elevation is measured toward the observer or the Sun, and nothing else."""

    with pytest.raises(ValueError, match="invalid elevation direction: 'north'"):
        bp.ring_elevation(RING, direction='north')


def test_ring_incidence_angle_rejects_an_unknown_pole(bp: Backplane) -> None:
    """The pole must be one of the four the docstring lists."""

    with pytest.raises(ValueError, match="invalid incidence angle pole: 'south'"):
        bp.ring_incidence_angle(RING, pole='south')


def test_ring_emission_angle_rejects_an_unknown_pole(bp: Backplane) -> None:
    """The pole must be one of the four the docstring lists."""

    with pytest.raises(ValueError, match="invalid emission angle pole: 'south'"):
        bp.ring_emission_angle(RING, pole='south')


def test_ring_angular_resolution_rejects_unknown_units(bp: Backplane) -> None:
    """The angular resolution is reported in radians or in kilometers."""

    with pytest.raises(ValueError, match="invalid units: 'deg'"):
        bp.ring_angular_resolution(RING, units='deg')


@pytest.mark.parametrize('name', ['ring_sub_observer_longitude',
                                  'ring_sub_solar_longitude'])
def test_a_ring_sub_point_longitude_rejects_an_unknown_reference(name: str,
                                                                 bp: Backplane) -> None:
    """The reference must be one of the six the docstring lists."""

    with pytest.raises(ValueError, match="invalid longitude reference: 'noon'"):
        getattr(bp, name)(RING, reference='noon')


@pytest.mark.parametrize('name', ['ring_sub_observer_longitude',
                                  'ring_sub_solar_longitude'])
@pytest.mark.parametrize('reference', ['aries', 'node', 'obs', 'oha', 'sun', 'sha'])
def test_a_ring_sub_point_longitude_accepts_every_reference(name: str, reference: str,
                                                            bp: Backplane) -> None:
    """Each reference shifts the zero point of the sub-point longitude."""

    longitude = getattr(bp, name)(RING, reference=reference).vals

    assert 0. <= longitude < TWOPI


@pytest.mark.parametrize('name', ['ring_sub_observer_longitude',
                                  'ring_sub_solar_longitude'])
def test_a_ring_sub_point_anti_reference_is_half_a_turn_away(name: str,
                                                             bp: Backplane) -> None:
    """'oha' is half a turn from 'obs', and 'sha' half a turn from 'sun'."""

    method = getattr(bp, name)

    assert (method(RING, reference='obs').vals
            - method(RING, reference='oha').vals) % TWOPI \
        == pytest.approx(np.pi, abs=1.e-9)
    assert (method(RING, reference='sun').vals
            - method(RING, reference='sha').vals) % TWOPI \
        == pytest.approx(np.pi, abs=1.e-9)


def test_the_observed_emission_angle_never_exceeds_a_right_angle(bp: Backplane) -> None:
    """The observed face is the one turned toward the observer, by definition."""

    observed = _unmasked(bp.ring_emission_angle(RING, pole='observed'))

    assert np.all(observed <= HALFPI + 1.e-12)
    assert np.all(observed >= 0.)


def test_the_north_and_prograde_emission_angles_agree_for_a_prograde_ring(
        bp: Backplane) -> None:
    """Saturn's rings are prograde, so its north pole is its prograde pole."""

    north = _unmasked(bp.ring_emission_angle(RING, pole='north'))
    prograde = _unmasked(bp.ring_emission_angle(RING, pole='prograde'))

    assert np.all(north == prograde)


def test_saturns_rings_are_prograde(bp: Backplane) -> None:
    """The sense of the rings comes from the planet named in the event key."""

    assert not bp._ring_is_retrograde(Backplane.standardize_event_key(RING))


def test_the_sense_of_a_moons_rings_comes_from_its_parent(bp: Backplane) -> None:
    """A body with no colon in its name is looked up through its parent planet."""

    assert not bp._ring_is_retrograde(Backplane.standardize_event_key(MOON))


def test_ring_shadow_incidence_is_the_angle_at_the_shadowing_ring(bp: Backplane) -> None:
    """The incidence angle at the surface is the one the ring cast the shadow at."""

    incidence = bp.ring_shadow_incidence(PLANET, RING)

    assert np.all(_unmasked(incidence) >= 0.)
    assert np.all(_unmasked(incidence) <= HALFPI + 1.e-12)
    assert bp.ring_shadow_incidence(PLANET, RING) is incidence


def test_a_radial_mode_can_be_stacked_on_another(bp: Backplane) -> None:
    """A radial_mode key names a backplane that another mode can be applied to."""

    first_key = ('radial_mode', ANNULUS_KEY, 2, EPOCH, 500., 0., 1.e-6, 0., 0., 'node')
    first = bp.radial_mode(ANNULUS_KEY, 2, EPOCH, 500., 0., 1.e-6)

    second = bp.radial_mode(first_key, 3, EPOCH, 200., 0., 1.e-6)

    assert np.all(np.abs(_unmasked(second - first)) <= 200. + 1.e-9)
    assert bp.radial_mode(first_key, 3, EPOCH, 200., 0., 1.e-6) is second


def test_a_radial_mode_carries_the_radius_limits_of_its_backplane(bp: Backplane) -> None:
    """The rmin and rmax of the underlying ring_radius are applied to the shifted radius.
    """

    mode = bp.radial_mode(ANNULUS_KEY, 2, EPOCH, 500., 0., 1.e-6)
    values = _unmasked(mode)

    assert np.all(values >= 80000.)
    assert np.all(values <= 100000.)


def test_a_radial_mode_is_refused_on_a_backplane_that_is_not_a_radius(
        bp: Backplane) -> None:
    """A mode shifts a ring radius, so it needs a ring_radius backplane to shift."""

    with pytest.raises(ValueError, match='radial modes only apply to ring_radius'):
        bp.radial_mode(('limb_altitude', LIMB, None, None), 2, EPOCH, 500., 0., 1.e-6)

##########################################################################################
