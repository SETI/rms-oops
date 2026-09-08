##########################################################################################
# tests/surface/test_surface_hints.py: hints and groundtracks across the Surfaces
##########################################################################################

import numpy as np
import pytest

from polymath                      import Scalar, Vector3
from oops.config                   import LOGGING
from oops.gravity.oblategravity    import OblateGravity
from oops.surface.centricellipsoid import CentricEllipsoid
from oops.surface.ellipsoid        import Ellipsoid
from oops.surface.graphicellipsoid import GraphicEllipsoid
from oops.surface.limb             import Limb
from oops.surface.ringplane        import RingPlane
from oops.surface.spheroid         import Spheroid

# Saturn's equatorial, intermediate and polar radii, which give an ellipsoid enough
# asymmetry for the three longitude conventions to differ
REQ = 60268.
RMID = 54364.
RPOL = 50000.

# A point above the surface of that body, and an observer well outside it
POS = Vector3([(0.5 * REQ, 0.2 * RMID, 0.5 * RPOL)])
OBS = Vector3((4. * REQ, 0., 0.))


def _ellipsoids() -> dict[str, Ellipsoid]:
    """One instance of each Surface subclass that solves for a normal, keyed by name.

    Returns:
        dict[str, Ellipsoid]: The surfaces to test.
    """

    return {
        'Spheroid':         Spheroid('SSB', 'J2000', (REQ, RPOL)),
        'Ellipsoid':        Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL)),
        'CentricEllipsoid': CentricEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL)),
        'GraphicEllipsoid': GraphicEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL)),
    }


@pytest.mark.parametrize('name', sorted(_ellipsoids()))
def test_the_hint_from_one_call_gives_the_same_coordinates_in_the_next(name: str) -> None:
    """The coefficient p is handed back so a second call can start from it."""

    surface = _ellipsoids()[name]
    (lon, lat, p) = surface.coords_from_vector3(POS, hints=True)

    (lon2, lat2, p2) = surface.coords_from_vector3(POS, hints=p)

    assert lon2.vals == pytest.approx(lon.vals, abs=1.e-12)
    assert lat2.vals == pytest.approx(lat.vals, abs=1.e-12)
    assert p2 == p


@pytest.mark.parametrize('name', sorted(_ellipsoids()))
def test_the_groundtrack_of_a_position_lies_on_the_surface(name: str) -> None:
    """With groundtrack set, the point on the surface below the position is returned."""

    surface = _ellipsoids()[name]

    (_, _, _, track) = surface.coords_from_vector3(POS, hints=True, groundtrack=True)

    assert surface.coords_from_vector3(track, axes=3)[2].vals \
           == pytest.approx(0., abs=1.e-6)


@pytest.mark.parametrize('name', sorted(_ellipsoids()))
def test_the_normal_hands_back_the_hint_it_was_given(name: str) -> None:
    """A hint handed in is handed back, so it can be reused by the next call."""

    surface = _ellipsoids()[name]

    (perp, hints) = surface.normal(POS, hints='reused')

    assert hints == 'reused'
    assert perp.norm().vals > 0.


@pytest.mark.parametrize('name', sorted(_ellipsoids()))
def test_the_normal_intercept_hands_back_the_coefficient(name: str) -> None:
    """`intercept_normal_to` returns the coefficient along with the surface point."""

    surface = _ellipsoids()[name]

    (track, p) = surface.intercept_normal_to(POS, hints='reused')

    assert surface.coords_from_vector3(track, axes=3)[2].vals \
           == pytest.approx(0., abs=1.e-6)
    assert p == 'reused'


@pytest.mark.parametrize('name', sorted(_ellipsoids()))
def test_the_position_from_coordinates_hands_back_the_hint(name: str) -> None:
    """A hint handed in is handed back by the reverse conversion as well."""

    surface = _ellipsoids()[name]
    coords = surface.coords_from_vector3(POS, axes=3)

    (pos, hints) = surface.vector3_from_coords(coords, hints='reused')

    assert hints == 'reused'
    assert (pos - POS).norm().vals == pytest.approx(0., abs=1.e-6)


@pytest.mark.parametrize('name', sorted(_ellipsoids()))
def test_a_point_inside_the_smallest_radius_is_inside_the_body(name: str) -> None:
    """Every point closer than the polar radius is inside, whatever the shape."""

    surface = _ellipsoids()[name]

    assert surface.position_is_inside(Vector3((0.5 * RPOL, 0., 0.)))
    assert not surface.position_is_inside(Vector3((2. * REQ, 0., 0.)))


##########################################################################################
# The ring plane, which carries an elevation and radial limits
##########################################################################################

RING_RADII = (74000., 140000.)
ELEVATION = 100.


def _ringplane(**kwargs) -> RingPlane:
    """A ring plane offset from the equator, with radial limits.

    Parameters:
        kwargs: Overrides of the default constructor arguments.

    Returns:
        RingPlane: The surface.
    """

    args = {'origin': 'SSB', 'frame': 'J2000', 'radii': RING_RADII,
            'elevation': ELEVATION}
    args.update(kwargs)

    return RingPlane(**args)


def test_the_third_ring_coordinate_is_measured_from_the_elevation() -> None:
    """An offset ring plane reports z relative to itself, not to the equator."""

    surface = _ringplane()
    pos = Vector3([(1.e5, 0., ELEVATION)])

    assert surface.coords_from_vector3(pos, axes=3)[2] == Scalar(0.)


@pytest.mark.parametrize('method, args',
                         [('coords_from_vector3', (Vector3([(1.e5, 0., 100.)]),)),
                          ('normal', (Vector3([(1.e5, 0., 100.)]),)),
                          ('vector3_from_coords', ((Scalar(1.e5), Scalar(0.)),))])
def test_a_ring_plane_hands_back_the_hint_it_was_given(method: str, args) -> None:
    """A hint handed in is handed back, so it can be reused by the next call."""

    surface = _ringplane()

    assert getattr(surface, method)(*args, hints='reused')[-1] == 'reused'


def test_a_ring_plane_intercept_hands_back_the_hint_it_was_given() -> None:
    """The intercept returns the position, the light time, and then the hint."""

    surface = _ringplane()
    obs = Vector3((0., 0., 1.e7))
    los = Vector3([(1.e5, 0., -1.e7)])

    (_, _, hints) = surface.intercept(obs, los, hints='reused')

    assert hints == 'reused'


def test_an_intercept_outside_the_radial_limits_is_masked() -> None:
    """The intercept is defined only between the inner and outer radii."""

    surface = _ringplane()
    obs = Vector3((0., 0., 1.e7))
    los = Vector3([(1.e5, 0., -1.e7), (5.e5, 0., -1.e7)])

    (pos, t) = surface.intercept(obs, los)

    assert list(pos.mask) == [False, True]
    assert list(t.mask) == [False, True]


def test_a_normal_outside_the_radial_limits_is_masked() -> None:
    """The surface normal is defined only where the ring is."""

    surface = _ringplane()
    pos = Vector3([(1.e5, 0., 100.), (5.e5, 0., 100.)])

    assert list(surface.normal(pos).mask) == [False, True]


def test_a_velocity_outside_the_radial_limits_is_masked() -> None:
    """The orbital velocity is defined only where the ring is."""

    surface = _ringplane(gravity=OblateGravity.SATURN)
    pos = Vector3([(1.e5, 0., 100.), (5.e5, 0., 100.)])

    assert list(surface.velocity(pos).mask) == [False, True]


def test_a_ring_plane_with_gravity_orbits_at_the_keplerian_rate() -> None:
    """With a gravity field, the velocity is the local orbital velocity."""

    surface = RingPlane('SSB', 'J2000', gravity=OblateGravity.SATURN)
    radius = 1.e5
    pos = Vector3([(radius, 0., 0.)])

    speed = surface.velocity(pos).norm().vals[0]

    assert speed == pytest.approx(radius * OblateGravity.SATURN.n(radius), rel=1.e-9)


def test_a_ring_plane_without_gravity_does_not_rotate() -> None:
    """With no gravity field there is no velocity field, and so no rate to limit."""

    surface = RingPlane('SSB', 'J2000')

    assert surface.velocity(Vector3([(1.e5, 0., 0.)])).norm().vals[0] == 0.
    assert surface._max_rate is None



##########################################################################################
# The limb, which delegates its coordinate conversions to the surface below it
##########################################################################################

def _limb() -> Limb:
    """A limb around a triaxial ellipsoid.

    Returns:
        Limb: The limb surface.
    """

    return Limb(Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL)))


def test_a_limb_needs_an_ellipsoidal_ground_surface() -> None:
    """A limb hangs off a body, so its ground surface has spherical coordinates."""

    with pytest.raises(ValueError, match='requires an ellipsoidal ground surface'):
        Limb(RingPlane('SSB', 'J2000'))


@pytest.mark.parametrize('name', ['lon_to_centric', 'lon_from_centric',
                                  'lon_to_graphic', 'lon_from_graphic'])
def test_a_limb_converts_longitudes_through_its_ground_surface(name: str) -> None:
    """Each longitude conversion is the ground surface's own."""

    limb = _limb()
    lon = Scalar([0.5, 1.5, 2.5])

    assert getattr(limb, name)(lon) == getattr(limb._ground, name)(lon)


@pytest.mark.parametrize('name', ['lat_to_centric', 'lat_from_centric',
                                  'lat_to_graphic', 'lat_from_graphic'])
def test_a_limb_converts_latitudes_through_its_ground_surface(name: str) -> None:
    """Each latitude conversion is the ground surface's own."""

    limb = _limb()
    lat = Scalar([0.5, -0.3, 0.1])
    lon = Scalar([0.5, 1.5, 2.5])

    assert getattr(limb, name)(lat, lon) == getattr(limb._ground, name)(lat, lon)


def test_the_hint_from_one_limb_call_gives_the_same_coordinates_in_the_next() -> None:
    """The coefficient p is handed back so a second call can start from it."""

    limb = _limb()
    pos = Vector3([(1.2 * REQ, 0.1 * RMID, 0.3 * RPOL)])
    (lon, lat, p) = limb.coords_from_vector3(pos, obs=OBS, hints=True)

    (lon2, lat2, p2) = limb.coords_from_vector3(pos, obs=OBS, hints=p)

    assert lon2.vals == pytest.approx(lon.vals, abs=1.e-9)
    assert lat2.vals == pytest.approx(lat.vals, abs=1.e-9)
    assert p2 == p


def test_a_limb_position_from_coordinates_hands_back_hints_and_groundtrack() -> None:
    """The hint given is returned unchanged, and the groundtrack is on the body."""

    limb = _limb()
    pos = Vector3([(1.2 * REQ, 0.1 * RMID, 0.3 * RPOL)])
    coords = limb.coords_from_vector3(pos, obs=OBS, axes=3)

    (back, hints, track) = limb.vector3_from_coords(coords, obs=OBS, hints='reused',
                                                    groundtrack=True)

    assert hints == 'reused'
    assert (back - pos).norm().vals == pytest.approx(0., abs=1.e-6)
    assert limb._ground.coords_from_vector3(track, axes=3)[2].vals \
           == pytest.approx(0., abs=1.e-6)


def test_the_clock_angle_of_a_groundtrack_hands_back_the_hint_it_was_given() -> None:
    """A hint handed in is handed back by the clock-angle conversion too."""

    limb = _limb()
    pos = Vector3([(1.2 * REQ, 0.1 * RMID, 0.3 * RPOL)])
    track = limb._ground.intercept_normal_to(pos)

    (clock, hints) = limb.clock_from_groundtrack(track, OBS, hints='reused')

    assert hints == 'reused'
    assert np.all(np.isfinite(clock.vals))


def test_the_longitude_and_latitude_of_a_position_come_from_its_groundtrack() -> None:
    """`lonlat_from_vector3` gives the coordinates of the point below a position."""

    limb = _limb()
    pos = Vector3([(1.2 * REQ, 0.1 * RMID, 0.3 * RPOL)])

    (lon, lat, hints, track) = limb.lonlat_from_vector3(pos, hints='reused',
                                                        groundtrack=True)

    assert hints == 'reused'
    assert limb._ground.coords_from_vector3(track, axes=3)[2].vals \
           == pytest.approx(0., abs=1.e-6)
    assert (lon, lat) == limb._ground.coords_from_vector3(track)[:2]

##########################################################################################
# The convergence logging of the iterative solvers
##########################################################################################

@pytest.mark.parametrize('name, marker',
                         [('Spheroid', 'Spheroid.intercept_normal_to'),
                          ('Ellipsoid', 'Ellipsoid.intercept_normal_to')])
def test_the_iterations_of_a_normal_intercept_can_be_logged(
        name: str, marker: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of the solver reports the change it made."""

    surface = _ellipsoids()[name]

    LOGGING.on()
    try:
        surface.intercept_normal_to(Vector3([(0.9 * REQ, 0.2 * RMID, 0.5 * RPOL)]))
    finally:
        LOGGING.off()

    assert marker in capsys.readouterr().out


def test_the_iterations_of_a_limb_intercept_can_be_logged(
        capsys: pytest.CaptureFixture[str]) -> None:
    """The limb solver reports the change it made on each pass."""

    limb = _limb()
    los = Vector3([(-4. * REQ, 0.6 * REQ, 0.4 * REQ)])

    LOGGING.on()
    try:
        limb.intercept(OBS, los)
    finally:
        LOGGING.off()

    assert 'Limb.intercept: iter=' in capsys.readouterr().out


def test_the_iterations_of_a_limb_position_from_z_and_clock_can_be_logged(
        capsys: pytest.CaptureFixture[str]) -> None:
    """The reverse limb solver reports the change it made on each pass."""

    limb = _limb()

    LOGGING.on()
    try:
        cept = limb.intercept_from_z_clock(Scalar([1000., 2000.]), Scalar([1., 2.]), OBS)
    finally:
        LOGGING.off()

    assert 'Limb.intercept_from_z_clock(): iter=' in capsys.readouterr().out

    # The points found are at the elevations that were asked for
    assert limb.z_clock_from_intercept(cept, OBS)[0].vals \
           == pytest.approx([1000., 2000.], abs=1.e-6)

##########################################################################################
