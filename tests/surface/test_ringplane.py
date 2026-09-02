##########################################################################################
# tests/surface/test_ringplane.py
##########################################################################################

import numpy as np
import pytest

from polymath               import Scalar, Vector3
from oops.constants         import TWOPI
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
