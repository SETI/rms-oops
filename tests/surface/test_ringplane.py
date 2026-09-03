##########################################################################################
# tests/surface/test_ringplane.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath               import Scalar, Vector3
from oops.constants         import TWOPI
from oops.gravity           import Gravity
from oops.event            import Event
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.ringplane import RingPlane


def test_ringplane():
    from oops.gravity import Gravity
    from oops.event import Event

    np.random.seed(8829)

    plane = RingPlane(Path.SSB, Frame.J2000)

    # Coordinate/vector conversions
    obs = np.random.rand(2,4,3,3)

    (r,theta,z) = plane.coords_from_vector3(obs,axes=3)
    assert (theta >= 0.).all()
    assert (theta < TWOPI).all()
    assert (r >= 0.).all()

    test = plane.vector3_from_coords((r,theta,z))
    assert np.all(np.abs(test.vals - obs) < 1.e-15)

    # Ring intercepts
    los = np.random.rand(2,4,3,3)
    obs[...,2] =  np.abs(obs[...,2])
    los[...,2] = -np.abs(los[...,2])

    (pts, factors) = plane.intercept(obs, los)
    assert abs(pts.to_scalar(2)).max() < 1.e-15

    angles = pts - obs
    assert (angles.sep(los) > -1.e-12).all()
    assert (angles.sep(los) <  1.e-12).all()

    # Intercepts that point away from the ring plane
    assert np.all(factors.vals > 0.)

    ######################################################################################
    # Test of radial modes
    ######################################################################################

    # Coordinate/vector conversions
    refplane = RingPlane(Path.SSB, Frame.J2000)

    plane = RingPlane(Path.SSB, Frame.J2000,
                      modes=[(10, 1000., 0., 0.)], epoch=0.)

    obs = 10.e3 * np.random.rand(2,4,3,3)

    (a,theta,z) = plane.coords_from_vector3(obs, time=0., axes=3)
    test = plane.vector3_from_coords((a,theta,z), time=0.)
    assert np.all(np.abs(test.vals - obs) < 1.e-11)

    test = plane.vector3_from_coords((a,theta,z), time=1.e8)
    assert np.all(np.abs(test.vals - obs) < 1.e-11)

    plane = RingPlane(Path.SSB, Frame.J2000,
                      modes=[(10, 1000., 0., 2*np.pi/100.)], epoch=0.)

    obs = 10.e3 * np.random.rand(2,4,3,3)

    (a,theta,z) = plane.coords_from_vector3(obs, time=0., axes=3)
    test = plane.vector3_from_coords((a,theta,z), time=0.)
    assert np.all(np.abs(test.vals - obs) < 1.e-11)

    test = plane.vector3_from_coords((a,theta,z), time=100.)
    assert np.all(np.abs(test.vals - obs) < 1.e-11)

    # longitudes are the same in both maps
    (a0,theta0,z0) = refplane.coords_from_vector3(obs, time=0., axes=3)
    assert theta0 == theta

    # radial offsets are out of phase when time=50.
    diff1 = a - a0
    (a,theta,z) = plane.coords_from_vector3(obs, time=50., axes=3)
    diff2 = a - a0
    assert abs(diff1 + diff2).max() < 1.e-11

    ######################################################################################
    # Test of velocities
    ######################################################################################

    pos = 10.e3 * np.random.rand(200,3)
    pos[...,2] = 0.     # set Z-coordinate to zero
    pos = Vector3(pos)

    # No gravity, no modes
    refplane = RingPlane(Path.SSB, Frame.J2000)

    vels = refplane.velocity(obs)
    assert vels == (0.,0.,0.)

    # No gravity, motionless mode
    plane = RingPlane(Path.SSB, Frame.J2000,
                      modes=[(10, 1000., 0., 0.)], epoch=0.)

    # A ring with modes requires a time, though this mode does not move
    vels = plane.velocity(obs, time=0.)
    assert vels == (0.,0.,0.)

    # No gravity, modes (10 cycles, 100 km amplitude, period = 10,000 s)
    plane = RingPlane(Path.SSB, Frame.J2000,
                      modes=[(10, 100., 0., 2.*np.pi/1.e4)], epoch=0.)

    TIME = 0.
    (a0,theta0) = plane.coords_from_vector3(pos, time=TIME - 0.5)
    (a ,theta ) = plane.coords_from_vector3(pos, time=TIME)
    (a1,theta1) = plane.coords_from_vector3(pos, time=TIME + 0.5)
    assert theta == theta0
    assert theta == theta1

    vels = plane.velocity(pos, time=TIME)
    v_angular = vels.perp(pos)
    v_radial = vels - v_angular

    sign = v_radial.dot(pos).sign()
    speed2 = sign * v_radial.norm()
    speed1 = a0 - a1
    assert abs(speed1 - speed2).max() < 2.e-9

    # Gravity, no modes
    plane = RingPlane(Path.SSB, Frame.J2000, gravity=Gravity.SATURN)

    (a, theta) = plane.coords_from_vector3(pos)

    vels = plane.velocity(pos)
    sep = vels.sep(pos)
    assert abs(sep - np.pi/2.).max() < 1.e-14

    speed1 = vels.norm()
    rate = np.minimum(Gravity.SATURN.n(a.vals), plane._max_rate)
    diff = (a * rate - speed1) / speed1
    assert abs(diff).max() < 1.e-15

    ######################################################################################
    # coords_of_event, event_from_coords
    ######################################################################################

    plane = RingPlane(Path.SSB, Frame.J2000)

    pos = Vector3(np.random.rand(2,4,3,3))
    vel = Vector3(np.random.rand(2,4,3,3))
    pos.insert_deriv('t', vel)

    event = Event(0., pos, Path.SSB, Frame.J2000)
    coords = plane.coords_of_event(event)
    test = plane.event_at_coords(0., coords)

    assert np.all(np.abs(test.pos.vals - pos.vals) < 1.e-15)
    assert np.all(np.abs(test.vel.vals - vel.vals) < 1.e-15)

        ##################################################################################
        # Note: Additional unit testing is performed in orbitplane.py
        ##################################################################################


# One radial mode: two cycles around the ring, 10 km in amplitude, drifting slowly
MODES = [(2, 10., 0., 1.e-6)]


def _ringed():
    """A RingPlane carrying a radial mode.

    Returns:
        RingPlane: The ring, whose radius therefore varies with time.
    """

    return RingPlane('SSB', 'J2000', modes=MODES, epoch=0.)


def test_radial_modes_require_a_time() -> None:
    """A ring whose radius varies with time cannot be evaluated without one.

    There is no sensible value to assume, so each entry point refuses rather than
    silently picking one.
    """

    ringed = _ringed()
    pos = Vector3((1.e5, 2.e4, 0.))

    with pytest.raises(ValueError, match='requires a time'):
        ringed.coords_from_vector3(pos)

    with pytest.raises(ValueError, match='requires a time'):
        ringed.vector3_from_coords((Scalar(1.e5), Scalar(0.2)))

    with pytest.raises(ValueError, match='requires a time'):
        ringed.velocity(pos)


def test_a_ring_without_modes_needs_no_time() -> None:
    """Time is irrelevant without modes, so it stays optional there."""

    plain = RingPlane('SSB', 'J2000')
    pos = Vector3((1.e5, 2.e4, 0.))

    assert plain.coords_from_vector3(pos)[0].vals > 0.


def test_radial_modes_shift_the_radius_with_time() -> None:
    """Given a time, the mode displaces the radius from the unmodulated value."""

    ringed = _ringed()
    plain = RingPlane('SSB', 'J2000')
    pos = Vector3((1.e5, 2.e4, 0.))

    modulated = ringed.coords_from_vector3(pos, time=Scalar(500.))[0]
    unmodulated = plain.coords_from_vector3(pos, time=Scalar(500.))[0]

    assert abs(modulated - unmodulated) > 1.
    assert abs(modulated - unmodulated) <= 10.      # bounded by the amplitude


def test_photon_to_coords_re_evaluates_a_time_dependent_ring() -> None:
    """The intercept must sit where the coordinates are at the converged surface time.

    The solver evaluates vector3_from_coords once, before iterating, for a non-virtual
    surface whose shape is fixed. A ring carrying a radial mode is not fixed, so it has
    to be re-evaluated as the surface time is refined; otherwise the returned event holds
    the position the coordinates occupied at the initial time guess.
    """

    # A large, fast mode, so the shape moves appreciably over the light travel time.
    ring = RingPlane('SSB', 'J2000', modes=[(2, 500., 0., 1.e-3)], epoch=0.)
    assert ring.IS_TIME_DEPENDENT is True
    assert ring.IS_VIRTUAL is False

    obs = Event(Scalar([0., 50.]), Vector3([(1.e6, 0., 1.e5), (1.e6, 0., 1.e5)]),
                Path.SSB, Frame.J2000)
    coords = (Scalar([1.4e5, 1.4e5]), Scalar([1.0, 2.0]))

    (surface_event, _) = ring.photon_to_coords(obs, coords)

    where_they_are = ring.vector3_from_coords(coords, time=surface_event.time)
    error = (surface_event.state - where_they_are).norm()

    assert float(error.max()) == pytest.approx(0., abs=1.e-6)


def test_photon_to_coords_still_solves_a_ring_without_modes() -> None:
    """A ring of fixed shape is solved the same way, its one evaluation being enough."""

    ring = RingPlane('SSB', 'J2000')
    assert ring.IS_TIME_DEPENDENT is False

    obs = Event(Scalar([0., 50.]), Vector3([(1.e6, 0., 1.e5), (1.e6, 0., 1.e5)]),
                Path.SSB, Frame.J2000)
    coords = (Scalar([1.4e5, 1.4e5]), Scalar([1.0, 2.0]))

    (surface_event, _) = ring.photon_to_coords(obs, coords)

    where_they_are = ring.vector3_from_coords(coords, time=surface_event.time)
    error = (surface_event.state - where_they_are).norm()

    assert float(error.max()) == pytest.approx(0., abs=1.e-6)


##########################################################################################
# Radial limits, elevation, and the local velocity field
##########################################################################################

RADII = (70000., 140000.)
INSIDE = Vector3((100000., 0., 0.))


def test_coords_are_cylindrical() -> None:
    """The coordinates are the radius and longitude in the ring plane."""

    surface = RingPlane('SSB', 'J2000')
    (radius, longitude) = surface.coords_from_vector3(INSIDE)

    assert radius == Scalar(100000.)
    assert longitude == Scalar(0.)


def test_a_third_coordinate_is_the_elevation() -> None:
    """axes=3 adds the vertical distance above the ring plane."""

    surface = RingPlane('SSB', 'J2000')
    (_, _, z) = surface.coords_from_vector3(Vector3((100000., 0., 250.)), axes=3)

    assert z == Scalar(250.)


def test_longitude_increases_in_the_prograde_direction() -> None:
    """A point on the +Y axis is a quarter turn from one on the +X axis."""

    surface = RingPlane('SSB', 'J2000')
    (_, longitude) = surface.coords_from_vector3(Vector3((0., 100000., 0.)))

    assert longitude.vals == pytest.approx(TWOPI / 4.)


def test_coords_and_vector3_are_inverses() -> None:
    """Converting a position to coordinates and back returns the position."""

    surface = RingPlane('SSB', 'J2000')
    pos = Vector3((60000., 80000., 500.))
    coords = surface.coords_from_vector3(pos, axes=3)

    assert surface.vector3_from_coords(coords).vals == pytest.approx(pos.vals)


def test_vector3_from_two_coordinates_lies_in_the_plane() -> None:
    """With no elevation given, the point sits on the ring plane."""

    surface = RingPlane('SSB', 'J2000')
    pos = surface.vector3_from_coords((Scalar(100000.), Scalar(0.)))

    assert pos == Vector3((100000., 0., 0.))


def test_radial_limits_mask_the_points_outside() -> None:
    """A radius outside the nominal limits is masked."""

    surface = RingPlane('SSB', 'J2000', radii=RADII)

    assert surface.coords_from_vector3(Vector3((10000., 0., 0.)))[0].mask
    assert surface.coords_from_vector3(Vector3((200000., 0., 0.)))[0].mask
    assert not surface.coords_from_vector3(INSIDE)[0].mask


def test_an_unbounded_ring_masks_nothing() -> None:
    """Without radii, the plane extends indefinitely."""

    surface = RingPlane('SSB', 'J2000')

    assert not surface.coords_from_vector3(Vector3((10000., 0., 0.)))[0].mask


def test_the_normal_is_the_z_axis() -> None:
    """The ring plane is the (x,y) plane, so its normal points along +Z."""

    assert RingPlane('SSB', 'J2000').normal(INSIDE).unit() == Vector3((0., 0., 1.))


def test_a_bounded_ring_has_a_normal() -> None:
    """Radial limits restrict where the surface is, not which way it faces."""

    surface = RingPlane('SSB', 'J2000', radii=RADII)

    assert surface.normal(INSIDE).unit() == Vector3((0., 0., 1.))


def test_a_bounded_ring_can_be_intercepted() -> None:
    """A line of sight crossing the ring plane meets it once."""

    surface = RingPlane('SSB', 'J2000', radii=RADII)
    (pos, t) = surface.intercept(Vector3((0., 0., 1.e6)), Vector3((0.1, 0., -1.)))

    assert pos.vals[2] == pytest.approx(0.)
    assert t > 0.


def test_an_unbounded_ring_can_be_intercepted() -> None:
    """The intercept lies where the line of sight crosses z = 0."""

    surface = RingPlane('SSB', 'J2000')
    (pos, t) = surface.intercept(Vector3((0., 0., 1.e6)), Vector3((0.1, 0., -1.)))

    assert pos.vals[2] == pytest.approx(0.)
    assert pos.vals[0] == pytest.approx(1.e5)
    assert t.vals == pytest.approx(1.e6)


def test_an_elevated_ring_is_offset_from_the_equator() -> None:
    """The elevation offsets the plane along the rotation axis."""

    surface = RingPlane('SSB', 'J2000', elevation=250.)
    (pos, _) = surface.intercept(Vector3((0., 0., 1.e6)), Vector3((0., 0., -1.)))

    assert pos.vals[2] == pytest.approx(250.)


def test_a_ring_without_gravity_has_no_velocity() -> None:
    """The velocity field is defined only when a gravity model is given."""

    assert RingPlane('SSB', 'J2000').velocity(INSIDE) == Vector3((0., 0., 0.))


def test_a_ring_with_gravity_orbits_the_center() -> None:
    """Particles move on circular Keplerian orbits about the central body."""

    surface = RingPlane('SSB', 'J2000', gravity=Gravity.lookup('SATURN'))
    velocity = surface.velocity(INSIDE)

    # A particle on the +X axis moves in the +Y direction, perpendicular to its radius
    assert velocity.vals[0] == pytest.approx(0., abs=1.e-9)
    assert velocity.vals[1] > 0.
    assert velocity.vals[2] == pytest.approx(0., abs=1.e-9)


def test_the_orbital_speed_falls_with_radius() -> None:
    """Keplerian motion is slower further out."""

    surface = RingPlane('SSB', 'J2000', gravity=Gravity.lookup('SATURN'))

    inner = abs(surface.velocity(Vector3((80000., 0., 0.))))
    outer = abs(surface.velocity(Vector3((140000., 0., 0.))))

    assert inner > outer


def test_ringplane_survives_a_pickle_round_trip() -> None:
    """Pickling restores the origin, frame, radii, and elevation."""

    surface = RingPlane('SSB', 'J2000', radii=RADII, elevation=250.)
    restored = pickle.loads(pickle.dumps(surface))

    assert isinstance(restored, RingPlane)
    assert restored.origin == surface.origin
    assert restored.frame == surface.frame
    assert restored.coords_from_vector3(INSIDE)[0] \
           == surface.coords_from_vector3(INSIDE)[0]

##########################################################################################
