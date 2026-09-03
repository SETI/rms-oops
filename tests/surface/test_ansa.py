##########################################################################################
# tests/surface/test_ansa.py
##########################################################################################

import pickle
from collections.abc import Iterator

import cspyce
import numpy as np
import pytest

from polymath               import Scalar, Vector3
from oops.body              import Body
from oops.frame             import Frame
from oops.gravity           import Gravity
from oops.path              import Path
from oops.surface.ansa      import Ansa
from oops.surface.ringplane import RingPlane
from programs.gold_master.test_support import TEST_SPICE_PREFIX

from oops.constants import PI, HALFPI
from tests.conftest import CORE_KERNELS


def test_ansa():
    np.random.seed(7742)

    surface = Ansa('SSB', 'J2000')

    # intercept()
    obs = Vector3( np.random.rand(10,3) * 1.e5)
    los = Vector3(-np.random.rand(10,3))

    (pos,t) = surface.intercept(obs, los)
    pos_xy = pos.element_mul((1,1,0))
    los_xy = los.element_mul((1,1,0))

    assert abs(pos_xy.sep(los_xy) - HALFPI).max() < 1.e-8
    assert abs(obs + t * los - pos).max() < 1.e-8

    # coords_from_vector3()
    obs = Vector3(np.random.rand(100,3) * 1.e6)
    pos = Vector3(np.random.rand(100,3) * 1.e5)

    (r,z) = surface.coords_from_vector3(pos, obs=obs, axes=2)

    pos_xy = pos.element_mul(Vector3((1,1,0)))
    pos_z  = pos.to_scalar(2)
    assert abs(pos_xy.norm() - abs(r)).max() < 1.e-8
    assert abs(pos_z - z).max() < 1.e-8

    (r,z,theta) = surface.coords_from_vector3(pos, obs=obs, axes=3)

    pos_xy = pos.element_mul(Vector3((1,1,0)))
    pos_z  = pos.to_scalar(2)
    assert abs(pos_xy.norm() - abs(r)).max() < 1.e-8
    assert abs(pos_z - z).max() < 1.e-8
    assert abs(theta).max() <= PI

    # vector3_from_coords()
    obs = Vector3(1.e-5 + np.random.rand(100,3) * 1.e6)
    r = Scalar(1.e-4 + np.random.rand(100) * 9e-4)
    z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
    theta = Scalar(np.random.rand(100))

    pos = surface.vector3_from_coords((r,z), obs=obs)

    pos_xy = pos.element_mul(Vector3((1,1,0)))
    pos_z  = pos.to_scalar(2)
    assert abs(pos_xy.norm() - abs(r)).max() < 1.e-8
    assert abs(pos_z - z).max() < 1.e-8

    obs_xy = obs.element_mul(Vector3((1,1,0)))
    assert abs(pos_xy.sep(obs_xy - pos_xy) - HALFPI).max() < 1.e-5

    pos1 = surface.vector3_from_coords((r,z,theta), obs=obs)
    pos1_xy = pos1.element_mul(Vector3((1,1,0)))
    assert abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5

    pos1 = surface.vector3_from_coords((r,z,-theta), obs=obs)
    pos1_xy = pos1.element_mul(Vector3((1,1,0)))
    assert abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5

    pos = surface.vector3_from_coords((-r,z), obs=obs)
    pos_xy = pos.element_mul(Vector3((1,1,0)))

    pos1 = surface.vector3_from_coords((-r,z,-theta), obs=obs)
    pos1_xy = pos1.element_mul(Vector3((1,1,0)))
    assert abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5

    pos1 = surface.vector3_from_coords((-r,z,theta), obs=obs)
    pos1_xy = pos1.element_mul(Vector3((1,1,0)))
    assert abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5

    # vector3_from_coords() & coords_from_vector3()
    obs = Vector3((1.e6,0,0))
    r = Scalar(1.e4 + np.random.rand(100) * 9.e4)
    r *= np.sign(2 * np.random.rand(100) - 1)
    z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
    theta = Scalar((2 * np.random.rand(100) - 1) * 1.)

    pos = surface.vector3_from_coords((r,z,theta), obs=obs)
    coords = surface.coords_from_vector3(pos, obs=obs, axes=3)
    assert abs(r - coords[0]).max() < 1.e-5
    assert abs(z - coords[1]).max() < 1.e-5
    assert abs(theta - coords[2]).max() < 1.e-8

    obs = Vector3(np.random.rand(100,3) * 1.e6)
    pos = Vector3(np.random.rand(100,3) * 1.e5)
    coords = surface.coords_from_vector3(pos, obs=obs, axes=3)
    test_pos = surface.vector3_from_coords(coords, obs=obs)
    assert abs(test_pos - pos).max() < 1.e-5

    # intercept() derivatives
    obs = Vector3(np.random.rand(10,3))
    obs.insert_deriv('obs', Vector3.IDENTITY)
    los = Vector3(-np.random.rand(10,3))
    los.insert_deriv('los', Vector3.IDENTITY)
    (pos0,t0) = surface.intercept(obs, los, derivs=True)

    eps = 1e-6
    (pos1,t1) = surface.intercept(obs + (eps,0,0), los, derivs=False)
    dpos_dobs_test = (pos1 - pos0) / eps
    dt_dobs_test = (t1 - t0) / eps
    assert abs(dpos_dobs_test - pos0.d_dobs.vals[...,0]).max() < 1.e-6
    assert abs(dt_dobs_test - t0.d_dobs.vals[...,0]).max() < 1.e-6

    (pos1,t1) = surface.intercept(obs + (0,eps,0), los, derivs=False)
    dpos_dobs_test = (pos1 - pos0) / eps
    dt_dobs_test = (t1 - t0) / eps
    assert abs(dpos_dobs_test - pos0.d_dobs.vals[...,1]).max() < 1.e-5
    assert abs(dt_dobs_test - t0.d_dobs.vals[...,1]).max() < 1.e-6

    (pos1,t1) = surface.intercept(obs + (0,0,eps), los, derivs=False)
    dpos_dobs_test = (pos1 - pos0) / eps
    dt_dobs_test = (t1 - t0) / eps
    assert abs(dpos_dobs_test - pos0.d_dobs.vals[...,2]).max() < 1.e-5
    assert abs(dt_dobs_test - t0.d_dobs.vals[...,2]).max() < 1.e-6

    eps = 1e-6
    (pos1,t1) = surface.intercept(obs, los + (eps,0,0), derivs=False)
    dpos_dlos_test = (pos1 - pos0) / eps
    dt_dlos_test = (t1 - t0) / eps
    assert abs(dpos_dlos_test - pos0.d_dlos.vals[...,0]).max() < 1.e-2
    assert abs(dt_dlos_test - t0.d_dlos.vals[...,0]).max() < 1.e-2

    (pos1,t1) = surface.intercept(obs, los + (0,eps,0), derivs=False)
    dpos_dlos_test = (pos1 - pos0) / eps
    dt_dlos_test = (t1 - t0) / eps
    assert abs(dpos_dlos_test - pos0.d_dlos.vals[...,1]).max() < 1.e-2
    assert abs(dt_dlos_test - t0.d_dlos.vals[...,1]).max() < 1.e-2

    (pos1,t1) = surface.intercept(obs, los + (0,0,eps), derivs=False)
    dpos_dlos_test = (pos1 - pos0) / eps
    dt_dlos_test = (t1 - t0) / eps
    assert abs(dpos_dlos_test - pos0.d_dlos.vals[...,2]).max() < 1.e-2
    assert abs(dt_dlos_test - t0.d_dlos.vals[...,2]).max() < 1.e-2

##########################################################################################
# Construction from a ring plane or a body, radial limits, hints, and serialization
##########################################################################################

RADII = (74000., 140000.)


@pytest.fixture(scope='module', autouse=True)
def _solar_system() -> Iterator[None]:
    """The solar system bodies, whose rings the constructors below refer to.

    Defining them is expensive, so one definition serves the whole module.
    """

    for path in TEST_SPICE_PREFIX.retrieve(CORE_KERNELS):
        cspyce.furnsh(path)

    Path._reset_caches()
    Frame._reset_caches()
    Body.reset_registry()
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2010-01-01')

    yield

    Path._reset_caches()
    Frame._reset_caches()
    Body.reset_registry()


def test_an_ansa_inherits_the_gravity_and_radii_of_a_ring_plane() -> None:
    """`for_ringplane` builds the ansa of a ring plane already in hand."""

    ringplane = RingPlane('SATURN', 'IAU_SATURN', radii=RADII,
                          gravity=Gravity.lookup('SATURN'))

    surface = Ansa.for_ringplane(ringplane)

    assert surface.ringplane is ringplane
    assert surface._gravity is ringplane._gravity
    assert list(surface._radii) == list(RADII)


def test_an_ansa_can_be_built_from_a_ring_body() -> None:
    """`for_body` uses a body whose own surface is a ring plane."""

    body = Body.lookup('SATURN_MAIN_RINGS')

    surface = Ansa.for_body(body)

    assert surface.ringplane is body.surface
    assert surface.origin is body.path.waypoint


def test_an_ansa_built_from_a_planet_uses_its_ring_body() -> None:
    """A body with a spherical surface defers to the ring body hanging off it."""

    planet = Body.lookup('SATURN')

    surface = Ansa.for_body(planet)

    assert surface.ringplane is planet.ring_body.surface


def test_an_explicit_gravity_overrides_the_one_from_the_ring_plane() -> None:
    """A gravity field given outright is used in place of the ring plane's."""

    ringplane = RingPlane('SATURN', 'IAU_SATURN')
    gravity = Gravity.lookup('JUPITER')

    surface = Ansa('SATURN', 'IAU_SATURN', ringplane=ringplane, gravity=gravity)

    assert surface._gravity is gravity


def test_radial_limits_mask_the_coordinates_outside_them() -> None:
    """A radius beyond the limits is masked, and so are the other two coordinates."""

    surface = Ansa('SSB', 'J2000', radii=RADII)
    pos = Vector3([(50000., 0., 1000.), (100000., 0., 1000.), (200000., 0., 1000.)])
    obs = Vector3((0., -1.e9, 0.))

    (r, z, theta) = surface.coords_from_vector3(pos, obs=obs, axes=3)

    assert list(r.mask) == [True, False, True]
    assert list(z.mask) == [True, False, True]
    assert list(theta.mask) == [True, False, True]


def test_the_unmasked_surface_drops_the_radial_limits() -> None:
    """The unmasked counterpart shares the geometry but keeps every radius."""

    surface = Ansa('SSB', 'J2000', radii=RADII)
    pos = Vector3([(50000., 0., 1000.)])
    obs = Vector3((0., -1.e9, 0.))

    assert surface.unmasked is not surface
    assert surface.unmasked._radii is None
    assert not np.any(surface.unmasked.coords_from_vector3(pos, obs=obs)[0].mask)


def test_an_unlimited_ansa_is_its_own_unmasked_surface() -> None:
    """With no radial limits there is nothing to unmask."""

    surface = Ansa('SSB', 'J2000')

    assert surface.unmasked is surface


def test_the_hints_are_passed_through_every_conversion() -> None:
    """A hint handed in is handed back, so it can be reused by the next call."""

    surface = Ansa('SSB', 'J2000')
    pos = Vector3([(100000., 0., 1000.)])
    obs = Vector3((0., -1.e9, 0.))
    los = pos - obs

    assert surface.coords_from_vector3(pos, obs=obs, hints=True)[-1] is True
    assert surface.vector3_from_coords((Scalar(1.e5), Scalar(1.e3)), obs=obs,
                                       hints=True)[-1] is True
    assert surface.intercept(obs, los, hints=True)[-1] is True
    assert surface.normal(pos, hints=True)[-1] is True


def test_the_normal_to_an_ansa_is_the_z_axis() -> None:
    """The ansa surface is a cylinder about the z-axis, so its normal is that axis."""

    surface = Ansa('SSB', 'J2000')
    pos = Vector3([(100000., 0., 1000.), (0., 50000., -2000.)])

    assert surface.normal(pos) == Vector3([(0., 0., 1.), (0., 0., 1.)])


def test_an_ansa_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the surface from its origin, frame, gravity and radii."""

    surface = Ansa('SATURN', 'IAU_SATURN', radii=RADII)
    pos = Vector3([(100000., 0., 1000.)])
    obs = Vector3((0., -1.e9, 0.))

    revived = pickle.loads(pickle.dumps(surface))

    assert list(revived._radii) == list(RADII)
    assert revived.origin is surface.origin
    assert revived.coords_from_vector3(pos, obs=obs) \
           == surface.coords_from_vector3(pos, obs=obs)

##########################################################################################
