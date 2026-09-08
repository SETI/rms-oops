##########################################################################################
# tests/backplane/test_geometry.py: resolution, ansa, limb, pole, orbit and pixel
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane
from oops.constants import TWOPI

PLANET = 'SATURN'
RING = 'SATURN:RING'
ANSA = 'SATURN:ANSA'
LIMB = 'SATURN:LIMB'
MOON = 'MIMAS'
HALFPI = np.pi / 2.


def _unmasked(array) -> np.ndarray:
    """The values of a backplane where it is not masked."""

    return array.vals[array.antimask]


##########################################################################################
# resolution
##########################################################################################

@pytest.mark.parametrize('axis', ['u', 'v'])
def test_resolution_is_positive(axis: str, bp: Backplane) -> None:
    """Resolution is a length per pixel, so it is never negative."""

    assert np.all(_unmasked(bp.resolution(PLANET, axis=axis)) > 0.)


def test_resolution_matches_the_pixel_scale_at_the_range(bp: Backplane) -> None:
    """Resolution is the angular pixel size times the distance to the surface."""

    resolution = bp.resolution(PLANET, axis='u')
    distance = bp.distance(PLANET)
    antimask = resolution.antimask
    pixel = 9.136e-05 / 20.

    expected = distance.vals[antimask] * pixel
    assert np.allclose(resolution.vals[antimask], expected, rtol=0.05)


def test_finest_resolution_is_no_coarser_than_the_coarsest(bp: Backplane) -> None:
    """The optimal direction resolves at least as finely as the worst."""

    finest = _unmasked(bp.finest_resolution(PLANET))
    coarsest = _unmasked(bp.coarsest_resolution(PLANET))

    assert np.all(finest <= coarsest + 1.e-9)


def test_the_axis_resolutions_lie_between_the_extremes(bp: Backplane) -> None:
    """The resolution along either image axis falls within the extreme values."""

    finest = bp.finest_resolution(PLANET)
    coarsest = bp.coarsest_resolution(PLANET)
    along_u = bp.resolution(PLANET, axis='u')
    antimask = along_u.antimask

    assert np.all(along_u.vals[antimask] >= finest.vals[antimask] - 1.e-9)
    assert np.all(along_u.vals[antimask] <= coarsest.vals[antimask] + 1.e-9)


def test_center_resolution_is_gridless(bp: Backplane) -> None:
    """The center resolution refers to the body's path, with no spatial extent."""

    assert bp.center_resolution(PLANET).shape == ()


def test_center_resolution_matches_the_disk(bp: Backplane) -> None:
    """The resolution at the center falls within the range across the disk."""

    values = _unmasked(bp.resolution(PLANET, axis='u'))
    center = bp.center_resolution(PLANET, axis='u')

    assert values.min() <= center.vals <= values.max()


##########################################################################################
# ansa
##########################################################################################

def test_ansa_radius_is_positive_by_default(bp: Backplane) -> None:
    """The 'positive' radius type reports every radius as a positive value."""

    assert np.all(_unmasked(bp.ansa_radius(ANSA)) >= 0.)


@pytest.mark.parametrize('radius_type', ['right', 'left'])
def test_ansa_radius_is_signed_on_the_two_sides(radius_type: str,
                                                bp: Backplane) -> None:
    """The 'right' and 'left' types distinguish the two ansae by sign."""

    radii = _unmasked(bp.ansa_radius(ANSA, radius_type=radius_type))

    assert np.any(radii > 0.)
    assert np.any(radii < 0.)


def test_left_and_right_ansa_radii_are_negatives(bp: Backplane) -> None:
    """'left' is the opposite of 'right'."""

    right = bp.ansa_radius(ANSA, radius_type='right')
    left = bp.ansa_radius(ANSA, radius_type='left')
    antimask = right.antimask

    assert np.allclose(right.vals[antimask], -left.vals[antimask])


def test_ansa_radius_limit_masks_the_rest(bp: Backplane) -> None:
    """rmax bounds the absolute value of the radius."""

    limited = bp.ansa_radius(ANSA, rmax=100000.)

    assert np.all(np.abs(_unmasked(limited)) <= 100000.)
    assert np.any(limited.mask)


def test_ansa_altitude_straddles_the_ring_plane(bp: Backplane) -> None:
    """The ansa surface reaches above and below the ring plane."""

    altitudes = _unmasked(bp.ansa_altitude(ANSA))

    assert altitudes.min() < 0.
    assert altitudes.max() > 0.


def test_ansa_longitude_spans_the_full_circle(bp: Backplane) -> None:
    """Longitudes are angles, reported between 0 and 2*pi."""

    longitudes = _unmasked(bp.ansa_longitude(ANSA))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


@pytest.mark.parametrize('reference', ['aries', 'node', 'obs', 'sun', 'oha', 'sha'])
def test_ansa_longitude_accepts_every_reference(reference: str, bp: Backplane) -> None:
    """Each reference shifts the zero point of longitude."""

    longitudes = _unmasked(bp.ansa_longitude(ANSA, reference=reference))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


@pytest.mark.parametrize('method', ['ansa_radial_resolution',
                                    'ansa_vertical_resolution'])
def test_ansa_resolutions_are_positive(method: str, bp: Backplane) -> None:
    """Resolution is a length per pixel, so it is never negative."""

    assert np.all(_unmasked(getattr(bp, method)(ANSA)) > 0.)


##########################################################################################
# limb
##########################################################################################

def test_limb_altitude_is_measured_from_the_surface(bp: Backplane) -> None:
    """The limb surface spans a range of altitudes above the globe."""

    altitudes = _unmasked(bp.limb_altitude(LIMB))

    assert altitudes.max() > 0.
    assert altitudes.size > 0


def test_limb_altitude_limits_mask_the_rest(bp: Backplane) -> None:
    """zmin and zmax restrict the backplane to a shell."""

    limited = bp.limb_altitude(LIMB, zmin=0., zmax=10000.)
    values = _unmasked(limited)

    assert np.all(values >= 0.)
    assert np.all(values <= 10000.)
    assert np.any(limited.mask)


def test_limb_latitude_stays_between_the_poles(bp: Backplane) -> None:
    """Latitude runs from -pi/2 at the south pole to +pi/2 at the north."""

    latitudes = _unmasked(bp.limb_latitude(LIMB))

    assert np.all(latitudes >= -HALFPI)
    assert np.all(latitudes <= HALFPI)


def test_limb_longitude_spans_the_full_circle(bp: Backplane) -> None:
    """Longitudes are angles, reported between 0 and 2*pi."""

    longitudes = _unmasked(bp.limb_longitude(LIMB))

    assert np.all(longitudes >= 0.)
    assert np.all(longitudes < TWOPI)


def test_limb_clock_angle_is_an_angle(bp: Backplane) -> None:
    """The clock angle is measured around the sky, so it wraps at 2*pi."""

    angles = _unmasked(bp.limb_clock_angle(LIMB))

    assert np.all(angles >= -TWOPI)
    assert np.all(angles <= TWOPI)


##########################################################################################
# pole
##########################################################################################

def test_pole_angles_are_gridless(bp: Backplane) -> None:
    """A pole backplane refers to the body's path, with no spatial extent."""

    assert bp.pole_clock_angle(PLANET).shape == ()
    assert bp.pole_position_angle(PLANET).shape == ()


def test_pole_position_angle_is_the_complement_of_the_clock_angle(
        bp: Backplane) -> None:
    """One is measured clockwise on the sky and the other counterclockwise."""

    clock = bp.pole_clock_angle(PLANET).vals
    position = bp.pole_position_angle(PLANET).vals

    assert (clock + position) % TWOPI == pytest.approx(0., abs=1.e-9)


def test_pole_clock_angle_is_within_the_circle(bp: Backplane) -> None:
    """The angle is reported between 0 and 2*pi."""

    clock = bp.pole_clock_angle(PLANET).vals

    assert 0. <= clock < TWOPI


##########################################################################################
# orbit
##########################################################################################

def test_orbit_longitude_is_gridless(bp: Backplane) -> None:
    """An orbit backplane refers to the moon's path, with no spatial extent."""

    assert bp.orbit_longitude(MOON).shape == ()


def test_orbit_longitude_spans_the_full_circle(bp: Backplane) -> None:
    """Longitudes are angles, reported between 0 and 2*pi."""

    longitude = bp.orbit_longitude(MOON).vals

    assert 0. <= longitude < TWOPI


@pytest.mark.parametrize('reference', ['aries', 'node', 'obs', 'sun', 'oha', 'sha'])
def test_orbit_longitude_accepts_every_reference(reference: str,
                                                 bp: Backplane) -> None:
    """Each reference shifts the zero point of longitude."""

    longitude = bp.orbit_longitude(MOON, reference=reference).vals

    assert 0. <= longitude < TWOPI


def test_orbit_longitude_accepts_an_explicit_planet(bp: Backplane) -> None:
    """The central body defaults to the parent of the targeted body."""

    assert bp.orbit_longitude(MOON, planet=PLANET) == bp.orbit_longitude(MOON)


##########################################################################################
# pixel
##########################################################################################

def test_body_diameter_in_pixels_matches_the_intercepted_area(bp: Backplane) -> None:
    """Saturn is about 20 pixels across in this synthetic observation."""

    diameter = bp.body_diameter_in_pixels(PLANET).vals

    assert 15. < diameter < 25.


def test_body_diameter_is_gridless(bp: Backplane) -> None:
    """The diameter refers to the body's path, with no spatial extent."""

    assert bp.body_diameter_in_pixels(PLANET).shape == ()


@pytest.mark.parametrize('axis', ['u', 'v', 'min', 'max'])
def test_body_diameter_accepts_every_axis(axis: str, bp: Backplane) -> None:
    """Each axis gives a positive apparent diameter."""

    assert bp.body_diameter_in_pixels(PLANET, axis=axis).vals > 0.


def test_a_larger_radius_gives_a_larger_diameter(bp: Backplane) -> None:
    """Overriding the radius scales the apparent diameter with it."""

    normal = bp.body_diameter_in_pixels(PLANET).vals
    doubled = bp.body_diameter_in_pixels(PLANET, radius=2. * 60268.).vals

    assert doubled > normal


@pytest.mark.parametrize('axis', ['u', 'v'])
def test_center_coordinate_falls_inside_the_field(axis: str, bp: Backplane) -> None:
    """The camera points at Saturn, so its center lies within the 40x40 grid."""

    coordinate = bp.center_coordinate(PLANET, axis=axis).vals

    assert 0. <= coordinate <= 40.


def test_center_coordinate_is_gridless(bp: Backplane) -> None:
    """The center coordinate refers to the body's path, with no spatial extent."""

    assert bp.center_coordinate(PLANET).shape == ()

##########################################################################################
# The caching, masking and validation shared by the ansa and limb backplanes
##########################################################################################

# A radius limit that leaves the innermost pixels of the ansa unmasked and masks the rest
ANSA_RMAX = 120000.

# Altitude limits in km, and the same limits as fractions of Saturn's largest radius
LIMB_ZMIN = 0.
LIMB_ZMAX = 10000.
LIMB_ZMAX_SCALED = 0.2


def test_a_backplane_is_evaluated_once_and_cached(bp: Backplane) -> None:
    """A second request returns the array already registered, not a new one."""

    assert bp.ansa_radius(ANSA) is bp.ansa_radius(ANSA)
    assert bp.ansa_altitude(ANSA) is bp.ansa_altitude(ANSA)
    assert bp.ansa_longitude(ANSA) is bp.ansa_longitude(ANSA)
    assert bp.ansa_radial_resolution(ANSA) is bp.ansa_radial_resolution(ANSA)
    assert bp.ansa_vertical_resolution(ANSA) is bp.ansa_vertical_resolution(ANSA)


def test_a_limb_backplane_is_evaluated_once_and_cached(bp: Backplane) -> None:
    """A second request returns the array already registered, not a new one."""

    assert bp.limb_altitude(LIMB) is bp.limb_altitude(LIMB)
    assert bp.limb_longitude(LIMB) is bp.limb_longitude(LIMB)
    assert bp.limb_latitude(LIMB) is bp.limb_latitude(LIMB)
    assert bp.limb_clock_angle(LIMB) is bp.limb_clock_angle(LIMB)


def test_ansa_radius_reports_left_and_right_as_opposites(bp: Backplane) -> None:
    """The "left" radius is the negative of the "right" one, pixel for pixel."""

    right = bp.ansa_radius(ANSA, radius_type='right')
    left = bp.ansa_radius(ANSA, radius_type='left')

    assert np.all(_unmasked(left + right) == 0.)


def test_ansa_radius_reports_a_positive_radius_on_both_sides(bp: Backplane) -> None:
    """The "positive" radius is the magnitude of the signed one."""

    signed = bp.ansa_radius(ANSA, radius_type='right')
    positive = bp.ansa_radius(ANSA, radius_type='positive')

    assert np.all(_unmasked(positive) >= 0.)
    assert np.all(_unmasked(positive - signed.abs()) == 0.)


def test_ansa_radius_rejects_an_unknown_radius_type(bp: Backplane) -> None:
    """The radius type has to be one of the three the docstring lists."""

    with pytest.raises(ValueError, match="invalid radius_type: 'inward'"):
        bp.ansa_radius(ANSA, radius_type='inward')


def test_ansa_longitude_rejects_an_unknown_reference(bp: Backplane) -> None:
    """The longitude reference has to be one of the six the docstring lists."""

    with pytest.raises(ValueError, match="invalid longitude reference: 'noon'"):
        bp.ansa_longitude(ANSA, reference='noon')


def test_a_radius_limit_masks_the_ansa_beyond_it(bp: Backplane) -> None:
    """rmax masks every pixel whose ansa radius exceeds it, in either direction."""

    limited = bp.ansa_radius(ANSA, rmax=ANSA_RMAX)

    assert limited.count_masked() > bp.ansa_radius(ANSA).count_masked()
    assert np.all(np.abs(_unmasked(limited)) <= ANSA_RMAX)


@pytest.mark.parametrize('name', ['ansa_altitude', 'ansa_longitude',
                                  'ansa_radial_resolution',
                                  'ansa_vertical_resolution'])
def test_an_ansa_backplane_inherits_the_mask_of_a_radius_backplane(name: str,
                                                                   bp: Backplane) -> None:
    """Given a radius backplane key, the array is remasked to match it."""

    radius_key = ('ansa_radius', ANSA, 'positive', ANSA_RMAX)
    limited = bp.ansa_radius(ANSA, rmax=ANSA_RMAX)

    remasked = getattr(bp, name)(radius_key)

    assert remasked.count_masked() == limited.count_masked()
    assert getattr(bp, name)(ANSA).count_masked() < limited.count_masked()


@pytest.mark.parametrize('name', ['limb_longitude', 'limb_latitude',
                                  'limb_clock_angle'])
def test_a_limb_backplane_inherits_the_mask_of_an_altitude_backplane(
        name: str, bp: Backplane) -> None:
    """Given a limb_altitude backplane key, the array is remasked to match it."""

    altitude_key = ('limb_altitude', LIMB, LIMB_ZMIN, LIMB_ZMAX)
    limited = bp.limb_altitude(LIMB, zmin=LIMB_ZMIN, zmax=LIMB_ZMAX)

    remasked = getattr(bp, name)(altitude_key)

    assert remasked.count_masked() == limited.count_masked()
    assert getattr(bp, name)(LIMB).count_masked() < limited.count_masked()


def test_scaled_limb_limits_are_fractions_of_the_body_radius(bp: Backplane) -> None:
    """With scaled=True the limits are multiples of the body's largest radius."""

    radius = bp._get_body_and_modifier(LIMB)[0].surface.radii.max()

    scaled = bp.limb_altitude(LIMB, zmin=LIMB_ZMIN, zmax=LIMB_ZMAX_SCALED, scaled=True)
    absolute = bp.limb_altitude(LIMB, zmin=LIMB_ZMIN, zmax=LIMB_ZMAX_SCALED * radius)

    assert scaled.count_masked() == absolute.count_masked()
    assert np.all(_unmasked(scaled) == _unmasked(absolute))


def test_limb_geometry_is_refused_on_a_surface_that_is_not_a_limb(bp: Backplane) -> None:
    """The limb backplanes need a limb surface, not the globe itself."""

    with pytest.raises(ValueError, match='invalid coordinate type for limb geometry'):
        bp._fill_limb_intercepts(Backplane.standardize_event_key(PLANET))


@pytest.mark.parametrize('name', ['_fill_ansa_intercepts', '_fill_ansa_longitudes'])
def test_ansa_geometry_is_refused_on_a_surface_that_is_not_cylindrical(
        name: str, bp: Backplane) -> None:
    """The ansa backplanes need a cylindrical surface, not the globe itself."""

    with pytest.raises(ValueError, match='invalid coordinate type for ansa geometry'):
        getattr(bp, name)(Backplane.standardize_event_key(PLANET))

##########################################################################################
# The argument checks and the order-dependent fills of the remaining backplane modules
##########################################################################################

# Uranus's rings rotate retrograde, so its north pole is not its prograde pole
RETROGRADE_RING = 'URANUS:RING'


@pytest.mark.parametrize('name', ['resolution', 'center_resolution'])
def test_a_resolution_rejects_an_axis_that_is_not_u_or_v(name: str,
                                                         bp: Backplane) -> None:
    """Resolution is measured along one image axis or the other."""

    with pytest.raises(ValueError, match="invalid axis: 'w'"):
        getattr(bp, name)(PLANET, axis='w')


def test_body_diameter_in_pixels_rejects_an_unknown_axis(bp: Backplane) -> None:
    """A diameter is measured along u, v, or the smaller or larger of the two."""

    with pytest.raises(ValueError, match="invalid axis: 'w'"):
        bp.body_diameter_in_pixels(PLANET, axis='w')


def test_center_coordinate_rejects_an_unknown_axis(bp: Backplane) -> None:
    """The center coordinate is measured along u or v."""

    with pytest.raises(ValueError, match="invalid axis: 'min'"):
        bp.center_coordinate(PLANET, axis='min')


@pytest.mark.parametrize('name', ['distance', 'light_time'])
def test_a_distance_rejects_an_unknown_photon_direction(name: str,
                                                        bp: Backplane) -> None:
    """A photon is either arriving or departing."""

    with pytest.raises(ValueError, match="invalid photon direction: 'both'"):
        getattr(bp, name)(PLANET, direction='both')


def test_orbit_longitude_rejects_an_unknown_reference(bp: Backplane) -> None:
    """The reference must be one of the six the docstring lists."""

    with pytest.raises(ValueError, match="invalid longitude reference: 'noon'"):
        bp.orbit_longitude(MOON, reference='noon')


def test_the_coarsest_resolution_can_be_the_first_one_asked_for(
        fresh_bp: Backplane) -> None:
    """The first request fills in the whole family of surface resolutions."""

    coarsest = fresh_bp.coarsest_resolution(PLANET)

    assert np.all(_unmasked(coarsest) >= _unmasked(fresh_bp.finest_resolution(PLANET)))


def test_declination_can_be_the_first_sky_backplane_asked_for(
        fresh_bp: Backplane) -> None:
    """The first request fills in both right ascension and declination."""

    declination = fresh_bp.declination()

    assert np.all(np.abs(_unmasked(declination)) <= HALFPI)
    assert fresh_bp.right_ascension().shape == declination.shape


def test_the_declination_of_a_surface_uses_its_arrival_directions(
        fresh_bp: Backplane) -> None:
    """Given a surface, the sky coordinates are those of the photons that lit it."""

    on_the_planet = fresh_bp.declination(PLANET)

    assert on_the_planet.count_masked() > 0
    assert np.all(np.abs(_unmasked(on_the_planet)) <= HALFPI)


def test_the_celestial_east_angle_is_filled_in_once(fresh_bp: Backplane) -> None:
    """The first request builds the sky derivatives; a second returns the array."""

    angle = fresh_bp.celestial_east_angle()

    assert np.all(np.abs(_unmasked(angle)) <= TWOPI)
    assert fresh_bp.celestial_east_angle() is angle


def test_the_center_declination_can_be_the_first_one_asked_for(
        fresh_bp: Backplane) -> None:
    """The first request fills in both center coordinates."""

    declination = fresh_bp.center_declination(PLANET)

    assert abs(declination.vals) <= HALFPI
    assert fresh_bp.center_right_ascension(PLANET).shape == declination.shape


def test_border_atop_is_evaluated_once_and_cached(bp: Backplane) -> None:
    """A second request returns the array already registered, not a new one."""

    key = ('ring_radius', RING, None, None)

    assert bp.border_atop(key, 100000.) is bp.border_atop(key, 100000.)


@pytest.mark.parametrize('name', ['lambert_law', 'lommel_seeliger_law'])
def test_a_photometric_law_is_evaluated_once_and_cached(name: str,
                                                        bp: Backplane) -> None:
    """A second request returns the array already registered, not a new one."""

    assert getattr(bp, name)(PLANET) is getattr(bp, name)(PLANET)


def test_where_sunward_on_a_ring_uses_the_observed_face(bp: Backplane) -> None:
    """A ring has two faces, so the incidence angle is measured from the observed one."""

    sunward = bp.where_sunward(RING)
    incidence = bp.ring_incidence_angle(RING, pole='observed')

    assert np.all(sunward.vals[incidence.antimask]
                  == (incidence.vals[incidence.antimask] <= HALFPI))


def test_where_inside_needs_a_key_naming_one_surface(bp: Backplane) -> None:
    """The interior test applies to one body lit by one source."""

    with pytest.raises(ValueError,
                       match='invalid event key for inside/outside calculations'):
        bp.where_inside((PLANET, RING), PLANET)


def test_nothing_is_inside_a_surface_that_has_no_interior(bp: Backplane) -> None:
    """A ring plane has no interior, so no point on the planet lies inside it."""

    assert not np.any(bp.where_inside(PLANET, RING).vals)
    assert np.any(bp.where_outside(PLANET, RING).vals)


def test_ansa_altitude_can_be_the_first_ansa_backplane_asked_for(
        fresh_bp: Backplane) -> None:
    """The first request fills in the whole family of ansa intercepts."""

    altitude = fresh_bp.ansa_altitude(ANSA)

    assert altitude.shape == fresh_bp.ansa_radius(ANSA).shape


def test_ring_longitude_can_be_the_first_ring_backplane_asked_for(
        fresh_bp: Backplane) -> None:
    """The first request fills in the whole family of ring intercepts."""

    longitude = fresh_bp.ring_longitude(RING)

    assert np.all(_unmasked(longitude) >= 0.)
    assert np.all(_unmasked(longitude) < TWOPI)


def test_the_prograde_incidence_angle_can_be_the_first_one_asked_for(
        fresh_bp: Backplane) -> None:
    """The prograde pole is the one the other three are derived from."""

    prograde = fresh_bp.ring_incidence_angle(RING, pole='prograde')

    assert np.all(_unmasked(prograde) >= 0.)
    assert np.all(_unmasked(prograde) <= np.pi)


@pytest.mark.parametrize('name', ['ring_incidence_angle', 'ring_emission_angle'])
def test_the_north_pole_of_a_retrograde_ring_is_not_its_prograde_pole(
        name: str, bp: Backplane) -> None:
    """Uranus's rings rotate retrograde, so the two poles are opposite."""

    north = getattr(bp, name)(RETROGRADE_RING, pole='north')
    prograde = getattr(bp, name)(RETROGRADE_RING, pole='prograde')

    assert np.all(np.abs(north.vals + prograde.vals - np.pi) < 1.e-12)


@pytest.mark.parametrize('name', ['ring_shadow_radius', 'ring_radius_in_front'])
def test_a_ring_shadow_backplane_is_evaluated_once_and_cached(name: str,
                                                              bp: Backplane) -> None:
    """A second request returns the array already registered, not a new one."""

    assert getattr(bp, name)(PLANET, RING) is getattr(bp, name)(PLANET, RING)


def test_ring_intercepts_are_refused_on_a_surface_that_is_not_a_ring(
        bp: Backplane) -> None:
    """The ring backplanes need a polar surface, not the globe itself."""

    with pytest.raises(ValueError, match='invalid coordinate type for ring geometry'):
        bp._fill_ring_intercepts(Backplane.standardize_event_key(PLANET))


def test_the_squashed_latitude_can_be_the_first_one_asked_for(
        fresh_bp: Backplane) -> None:
    """The squashed latitude is the one the intercepts are filled in with."""

    squashed = fresh_bp.latitude(PLANET, lat_type='squashed')

    assert np.all(np.abs(_unmasked(squashed)) <= HALFPI)


@pytest.mark.parametrize('name', ['sub_observer_longitude', 'sub_solar_longitude'])
def test_a_sub_point_longitude_in_its_default_form_is_returned_directly(
        name: str, fresh_bp: Backplane) -> None:
    """The IAU eastward longitude from zero is the form the others are derived from."""

    default = getattr(fresh_bp, name)(PLANET, reference='iau', direction='east',
                                      minimum=0)

    assert 0. <= default.vals < TWOPI

##########################################################################################
