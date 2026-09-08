##########################################################################################
# tests/surface/test_limb.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath              import Scalar, Vector3
from oops.frame.frame_     import Frame
from oops.path.path_       import Path
from oops.surface.limb     import Limb
from oops.surface.spheroid import Spheroid


def test_limb():
    from oops.surface.centricellipsoid import CentricEllipsoid
    from oops.surface.centricspheroid  import CentricSpheroid
    from oops.surface.ellipsoid        import Ellipsoid
    from oops.surface.graphicellipsoid import GraphicEllipsoid
    from oops.surface.graphicspheroid  import GraphicSpheroid
    from oops.surface.spheroid         import Spheroid
    from polymath                      import Matrix3

    np.random.seed(6922)

    REQ  = 60268.
    RMID = 54364.
    RPOL = 50000.

    NPTS = 1000

    ground = Spheroid('SSB', 'J2000', (REQ, RPOL))
    limb = Limb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[...,0] = -4 *REQ
    los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

    # Check (z,clock) conversions
    (z, clock, track2) = limb.z_clock_from_intercept(cept, obs, groundtrack=True)

    assert (track2 - track).norm().median() < 1.e-10
    assert (abs(track.element_mul(limb._ground._unsquash).norm()
                - limb._ground._req).median() < 1.e-10)
    assert (abs(track2.element_mul(limb._ground._unsquash).norm()
                - limb._ground._req).median() < 1.e-10)

    matrix = Matrix3.twovec(-obs, 2, Vector3.ZAXIS, 0)
    (x,y,_) = (matrix * ground.normal(track)).to_scalars()
    assert abs(y.arctan2(x) % Scalar.TWOPI - clock).median() < 1.e-10

    (x,y,_) = (matrix * ground.normal(track2)).to_scalars()
    assert abs(y.arctan2(x) % Scalar.TWOPI - clock).max() < 1.e-12

    assert abs((cept - track).sep(los)  - Scalar.HALFPI).median() < 1.e-12
    assert abs((cept - track2).sep(los) - Scalar.HALFPI).median() < 1.e-12
    assert abs((cept - track).sep(limb._ground.normal(track))).median() < 1.e-12
    assert abs((cept - track2).sep(limb._ground.normal(track2))).median() < 1.e-12

    cept2 = limb.intercept_from_z_clock(z, clock, obs)
    (z2, clock2) = limb.z_clock_from_intercept(cept2, obs)

    # The two methods are inverses; measured over this grid, the worst case is 2e-10
    assert abs(z2 - z).max() < 1.e-8
    assert abs(clock2 - clock).max() < 1.e-12
    assert (cept2 - cept).norm().max() < 1.e-8

    # Validate solution
    (cept, t, track) = limb.intercept(obs, los, groundtrack=True)
    normal = limb.normal(track).unit()
    assert abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12

    normal2 = cept - track
    sep = (normal2.sep(normal) + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
    assert abs(sep).max() < 1.e-10

    # Validate (lon,lat) conversions without z
    lon = np.random.random(NPTS) * Scalar.TWOPI
    lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)

    pos = limb.vector3_from_coords((lon,lat))
    coords = limb.coords_from_vector3(pos, axes=3)

    assert abs(coords[0] - lon).max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2]).max() < 1.e-6

    clock = np.random.random(NPTS) * Scalar.TWOPI
    obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                               REQ * np.random.random(NPTS),
                               REQ * np.random.random(NPTS))

    # Validate (lon,lat) conversions with z
    z = np.random.random(NPTS) * 10000. - 100.
    pos = limb.vector3_from_coords((lon,lat,z))
    coords = limb.coords_from_vector3(pos, axes=3)

    assert abs(coords[0] - lon).max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2] - z).max() < 1.e-10

    clock = np.random.random(NPTS) * Scalar.TWOPI
    obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                               REQ * np.random.random(NPTS),
                               REQ * np.random.random(NPTS))

    # Validate clock angles
    track = limb.groundtrack_from_clock(clock, obs)
    clock2 = limb.clock_from_groundtrack(track, obs)
    track2 = limb.groundtrack_from_clock(clock2, obs)

    assert (track2 - track).norm().max() < 1.e-6

    dclock = (clock2 - clock + Scalar.PI) % Scalar.TWOPI - Scalar.PI
    assert abs(dclock).max() < 1.e-12

    # Intercept with derivs
    N = 1000
    obs = Vector3(REQ * (0.95 + np.random.rand(N,3)))
    los = Vector3(np.random.randn(N,3))
    mask = obs.dot(los) > 0
    los[mask] = -los[mask]

    obs.insert_deriv('obs', Vector3.IDENTITY)
    los.insert_deriv('los', Vector3.IDENTITY)

    (pos, t, hints, track) = limb.intercept(obs, los, derivs=True, hints=True,
                                            groundtrack=True)

    eps = 1.
    dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
    for i in range(3):
        (pos1, t1, _, track1) = limb.intercept(obs+dobs[i], los, derivs=False,
                                               guess=t.wod, hints=hints.wod,
                                               groundtrack=True)

        (pos2, t2, _, track2) = limb.intercept(obs-dobs[i], los, derivs=False,
                                               guess=t.wod, hints=hints.wod,
                                               groundtrack=True)
        dpos_dobs = (pos1 - pos2) / (2*eps)
        assert abs(dpos_dobs - pos.d_dobs.vals[...,i]).max() < 1.e-9

        dt_dobs = (t1 - t2) / (2*eps)
        assert abs(dt_dobs - t.d_dobs.vals[...,i]).max() < 1.e-9

        dtrack_dobs = (track1 - track2) / (2*eps)
        assert abs(dtrack_dobs - track.d_dobs.vals[...,i]).max() < 1.e-9

    eps = 1.e-7
    dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
    for i in range(3):
        (pos1, t1, _, track1) = limb.intercept(obs, los+dlos[i], derivs=False,
                                               guess=t.wod, hints=hints.wod,
                                               groundtrack=True)

        (pos2, t2, _, track2) = limb.intercept(obs, los-dlos[i], derivs=False,
                                               guess=t.wod, hints=hints.wod,
                                               groundtrack=True)
        dpos_dlos = (pos1 - pos2) / (2*eps)
        scale = dpos_dlos.norm().median()
        assert abs(dpos_dlos - pos.d_dlos.vals[...,i]).max() < scale * 3.e-8

        dt_dlos = (t1 - t2) / (2*eps)
        scale = dt_dlos.abs().median()
        assert abs(dt_dlos - t.d_dlos.vals[...,i]).max() < scale * 3.e-8

        dtrack_dlos = (track1 - track2) / (2*eps)
        scale = dtrack_dlos.norm().median()
        assert abs(dtrack_dlos - track.d_dlos.vals[...,i]).max() < scale * 3.e-8

    # intercept_from_z_clock with derivs
    N = 1000
    z = Scalar(REQ * (0.95 + np.random.rand(N)))
    clock = Scalar(np.random.randn(N)) * Scalar.TWOPI
    obs = Vector3(REQ * (1.95 + np.random.rand(N,3)))

    z.insert_deriv('z', Scalar.ONE)
    clock.insert_deriv('clock', Scalar.ONE)
    obs.insert_deriv('obs', Vector3.IDENTITY)

    (pos, track) = limb.intercept_from_z_clock(z, clock, obs, derivs=True,
                                               groundtrack=True)
    eps = 1.
    (pos1, track1) = limb.intercept_from_z_clock(z + eps, clock, obs,
                                                 derivs=False,
                                                 groundtrack=True)
    (pos2, track2) = limb.intercept_from_z_clock(z - eps, clock, obs,
                                                 derivs=False,
                                                 groundtrack=True)
    dpos_dz = (pos1 - pos2) / (2*eps)
    assert abs(dpos_dz - pos.d_dz).max() < 1.e-9

    dtrack_dz = (track1 - track2) / (2*eps)
    assert abs(dtrack_dz - track.d_dz).max() < 1.e-9

    eps = 1.e-6
    (pos1, track1) = limb.intercept_from_z_clock(z, clock+eps, obs,
                                                 derivs=False,
                                                 groundtrack=True)
    (pos2, track2) = limb.intercept_from_z_clock(z, clock-eps, obs,
                                                 derivs=False,
                                                 groundtrack=True)

    dpos_dclock = (pos1 - pos2) / (2*eps)
    scale = dpos_dclock.norm().median()
    assert abs(dpos_dclock - pos.d_dclock).max() < scale * 3.e-8

    dtrack_dclock = (track1 - track2) / (2*eps)
    scale = dtrack_dclock.norm().median()
    assert abs(dtrack_dclock - track.d_dclock).max() < scale * 3.e-8

    eps = 1.
    dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
    for i in range(3):
        (pos1, track1) = limb.intercept_from_z_clock(z, clock, obs+dobs[i],
                                                     derivs=False,
                                                     groundtrack=True)

        (pos2, track2) = limb.intercept_from_z_clock(z, clock, obs-dobs[i],
                                                     derivs=False,
                                                     groundtrack=True)
        dpos_dobs = (pos1 - pos2) / (2*eps)
        scale = dpos_dobs.norm().median()
        assert abs(dpos_dobs - pos.d_dobs.vals[...,i]).max() < scale * 1.e-9

        dtrack_dobs = (track1 - track2) / (2*eps)
        scale = dtrack_dobs.norm().median()
        assert abs(dtrack_dobs - track.d_dobs.vals[...,i]).max() < scale * 1.e-9

    ######################################################################################

    ground = Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
    limb = Limb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[:,:,0] = -4 *REQ
    los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
    normal = limb.normal(track)

    # Check (z,clock) conversions
    (z, clock, track2) = limb.z_clock_from_intercept(cept, obs, groundtrack=True)

    assert (track2 - track).norm().median() < 1.e-10
    assert (abs(track.element_mul(limb._ground._unsquash).norm()
                - limb._ground._req).median() < 1.e-10)
    assert (abs(track2.element_mul(limb._ground._unsquash).norm()
                - limb._ground._req).median() < 1.e-10)

    matrix = Matrix3.twovec(-obs, 2, Vector3.ZAXIS, 0)
    (x,y,_) = (matrix * normal).to_scalars()
    assert abs(y.arctan2(x) % Scalar.TWOPI - clock).median() < 1.e-10

    (x,y,_) = (matrix * limb.normal(track2)).to_scalars()
    assert abs(y.arctan2(x) % Scalar.TWOPI - clock).max() < 1.e-12

    assert abs((cept - track).sep(los)  - Scalar.HALFPI).median() < 1.e-12
    assert abs((cept - track2).sep(los) - Scalar.HALFPI).median() < 1.e-12
    assert abs((cept - track).sep(limb._ground.normal(track))).median() < 1.e-12
    assert abs((cept - track2).sep(limb._ground.normal(track2))).median() < 1.e-12

    cept2 = limb.intercept_from_z_clock(z, clock, obs)
    (z2, clock2) = limb.z_clock_from_intercept(cept2, obs)

    # The two methods are inverses; measured over this grid, the worst case is 2e-10
    assert abs(z2 - z).max() < 1.e-8
    assert abs(clock2 - clock).max() < 1.e-12
    assert (cept2 - cept).norm().max() < 1.e-8

    # Validate solution
    assert abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12

    normal2 = cept - track
    sep = (normal2.sep(normal) + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
    assert abs(sep).max() < 1.e-10

    # Validate (lon,lat) conversions
    lon = np.random.random(NPTS) * Scalar.TWOPI
    lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
    z = np.random.random(NPTS) * 10000.

    pos = limb.vector3_from_coords((lon,lat,z))
    coords = limb.coords_from_vector3(pos, axes=3)

    assert abs(coords[0] - lon).max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2] - z).max() < 1.e-6

    clock = np.random.random(NPTS) * Scalar.TWOPI
    obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                               REQ * np.random.random(NPTS),
                               REQ * np.random.random(NPTS))

    # Validate clock angles
    track = limb.groundtrack_from_clock(clock, obs)
    clock2 = limb.clock_from_groundtrack(track, obs)
    track2 = limb.groundtrack_from_clock(clock2, obs)

    assert (track2 - track).norm().max() < 1.e-6

    dclock = (clock2 - clock + Scalar.PI) % Scalar.TWOPI - Scalar.PI
    assert abs(dclock).max() < 1.e-12

    ######################################################################################

    ground = CentricSpheroid('SSB', 'J2000', (REQ, RPOL))
    limb = Limb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[:,:,0] = -4 *REQ
    los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
    normal = limb.normal(track)

    assert abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12

    lon = np.random.random(NPTS) * Scalar.TWOPI
    lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
    z = np.random.random(NPTS) * 10000.

    pos = limb.vector3_from_coords((lon,lat,z))
    coords = limb.coords_from_vector3(pos, axes=3)

    diffs = abs(coords[0] - lon)
    diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
    assert diffs.max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2] - z).max() < 1.e-6

    ######################################################################################

    ground = GraphicSpheroid('SSB', 'J2000', (REQ, RPOL))
    limb = Limb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[:,:,0] = -4 *REQ
    los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
    normal = limb.normal(track)

    assert abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12

    lon = np.random.random(NPTS) * Scalar.TWOPI
    lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
    z = np.random.random(NPTS) * 10000.

    pos = limb.vector3_from_coords((lon,lat,z))
    coords = limb.coords_from_vector3(pos, axes=3)

    diffs = abs(coords[0] - lon)
    diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
    assert diffs.max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2] - z).max() < 1.e-6

    ######################################################################################

    ground = CentricEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
    limb = Limb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[:,:,0] = -4 *REQ
    los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
    normal = limb.normal(track)

    assert abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12

    lon = np.random.random(NPTS) * Scalar.TWOPI
    lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
    z = np.random.random(NPTS) * 10000.

    pos = limb.vector3_from_coords((lon,lat,z))
    coords = limb.coords_from_vector3(pos, axes=3)

    diffs = abs(coords[0] - lon)
    diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
    assert diffs.max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2] - z).max() < 1.e-6

    ######################################################################################

    ground = GraphicEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
    limb = Limb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[:,:,0] = -4 *REQ
    los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
    normal = limb.normal(track)

    assert abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12

    lon = np.random.random(NPTS) * Scalar.TWOPI
    lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
    z = np.random.random(NPTS) * 10000.

    pos = limb.vector3_from_coords((lon,lat,z))
    coords = limb.coords_from_vector3(pos, axes=3)

    diffs = abs(coords[0] - lon)
    diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
    assert diffs.max() < 1.e-12
    assert abs(coords[1] - lat).max() < 1.e-12
    assert abs(coords[2] - z).max() < 1.e-6


def _spheroid_limb():
    """A Limb over a strongly oblate spheroid, where the two definitions of z diverge.

    Returns:
        Limb: A Limb over a Saturn-like spheroid, Rpol/Req = 0.83.
    """

    from oops.surface.spheroid import Spheroid

    return Limb(Spheroid('SSB', 'J2000', (60268., 50000.)))


def _known_z_points(limb, obs):
    """Limb points built from known (z, clock) values.

    Parameters:
        limb (Limb): The Limb surface.
        obs (Vector3): The observer position.

    Returns:
        tuple[Vector3, Scalar, Scalar]: `(pos, z, clock)`, the points and the values they
        were built from.
    """

    z = Scalar(np.array([0., 10., 100., 500., 1000., 2000.] * 7))
    clock = Scalar(np.repeat(np.linspace(0., 2.*np.pi, 7, endpoint=False), 6))

    return (limb.intercept_from_z_clock(z, clock, obs), z, clock)


def test_z_clock_from_intercept_inverts_intercept_from_z_clock():
    """z must survive the round trip.

    z is the perpendicular distance from the surface, not the difference of the two
    radii. For an oblate body those differ by more than 1% of z and agree only at z == 0,
    which is why a test confined to the limb itself cannot tell them apart.
    """

    limb = _spheroid_limb()
    obs = Vector3([4*60268., 0, 0])
    (pos, z, _) = _known_z_points(limb, obs)

    assert abs(limb.z_clock_from_intercept(pos, obs)[0] - z).max() < 1.e-8


def test_z_clock_from_intercept_recovers_the_clock_angle():
    limb = _spheroid_limb()
    obs = Vector3([4*60268., 0, 0])
    (pos, _, clock) = _known_z_points(limb, obs)

    assert abs(limb.z_clock_from_intercept(pos, obs)[1] - clock).max() < 1.e-12


def test_z_clock_from_intercept_agrees_with_coords_from_vector3():
    """The two methods must report the same z for the same point."""

    limb = _spheroid_limb()
    obs = Vector3([4*60268., 0, 0])
    (pos, _, _) = _known_z_points(limb, obs)

    z_from_coords = limb.coords_from_vector3(pos, obs=obs, axes=3)[2]

    assert abs(limb.z_clock_from_intercept(pos, obs)[0] - z_from_coords).max() < 1.e-8


def test_z_clock_from_intercept_works_with_hints():
    """The hints branch supplies the coefficient p rather than solving for it."""

    limb = _spheroid_limb()
    obs = Vector3([4*60268., 0, 0])
    (pos, z, _) = _known_z_points(limb, obs)

    (_, _, p) = limb.z_clock_from_intercept(pos, obs, hints=True)

    assert abs(limb.z_clock_from_intercept(pos, obs, hints=p)[0] - z).max() < 1.e-8


def test_limb_unmasked_is_a_limb():
    limb = Limb(_spheroid_limb().ground, limits=(0., 1000.))

    assert type(limb.unmasked) is Limb


def test_polarlimb_unmasked_is_a_polarlimb():
    """A PolarLimb must not hand back a Limb: the two report different coordinates under
    the same method names."""

    from oops.surface.polarlimb import PolarLimb

    limb = PolarLimb(_spheroid_limb().ground, limits=(0., 1000.))

    assert type(limb.unmasked) is PolarLimb


def test_unmasked_limb_carries_no_limits():
    limb = Limb(_spheroid_limb().ground, limits=(0., 1000.))

    assert limb.unmasked.limits is None


##########################################################################################
# Limb coordinates, limits, and the ground surface
##########################################################################################

GROUND_RADII = (6378., 6357.)
OBSERVER = Vector3((1.e6, 0., 0.))


def _limb(limits=None) -> Limb:
    """A Limb surface on an oblate spheroid centered on the SSB."""

    ground = Spheroid(Path.SSB, Frame.J2000, GROUND_RADII)

    return Limb(ground, limits=limits)


def test_limb_reports_its_ground_surface() -> None:
    """The limb is defined relative to a spheroid or ellipsoid."""

    assert isinstance(_limb().ground, Spheroid)


def test_a_limb_without_limits_is_unbounded() -> None:
    """With no limits, every elevation is allowed."""

    assert _limb().limits is None


def test_a_limb_records_its_limits() -> None:
    """The limits bound the vertical distance from the ground surface."""

    assert _limb(limits=(0., 1000.)).limits == (0., 1000.)


def test_the_third_coordinate_is_the_elevation() -> None:
    """z is the distance above the ground surface, measured along its normal."""

    pos = Vector3((0., 6500., 0.))
    (_, _, z) = _limb().coords_from_vector3(pos, obs=OBSERVER, axes=3)

    assert z.vals == pytest.approx(6500. - GROUND_RADII[0])


def test_the_first_two_coordinates_are_the_ground_point() -> None:
    """lon and lat locate the point on the surface beneath the limb point."""

    pos = Vector3((0., 6500., 0.))
    (lon, lat) = _limb().coords_from_vector3(pos, obs=OBSERVER)

    assert lon.vals == pytest.approx(np.pi / 2.)
    assert lat.vals == pytest.approx(0., abs=1.e-12)


def test_coords_and_vector3_are_inverses() -> None:
    """Converting a position to coordinates and back returns the position."""

    limb = _limb()
    pos = Vector3((0., 6500., 0.))
    coords = limb.coords_from_vector3(pos, obs=OBSERVER, axes=3)

    assert limb.vector3_from_coords(coords, obs=OBSERVER).vals \
           == pytest.approx(pos.vals, abs=1.e-6)


def test_coords_can_return_the_converged_coefficient() -> None:
    """hints=True appends the coefficient p relating the ground point to the position."""

    result = _limb().coords_from_vector3(Vector3((0., 6500., 0.)), obs=OBSERVER,
                                         axes=3, hints=True)

    assert len(result) == 4
    assert result[3].vals > 0.


def test_coords_can_return_the_groundtrack() -> None:
    """groundtrack=True appends the point on the ground surface."""

    result = _limb().coords_from_vector3(Vector3((0., 6500., 0.)), obs=OBSERVER,
                                         axes=3, groundtrack=True)

    assert len(result) == 4
    assert result[3].vals == pytest.approx([0., GROUND_RADII[0], 0.], abs=1.e-6)


def test_an_elevation_above_the_limits_is_masked() -> None:
    """A z outside the limits is masked."""

    limb = _limb(limits=(0., 1000.))
    (_, _, z) = limb.coords_from_vector3(Vector3((0., 20000., 0.)), obs=OBSERVER, axes=3)

    assert z.mask


def test_an_elevation_within_the_limits_is_kept() -> None:
    """A z inside the limits is unmasked."""

    limb = _limb(limits=(0., 1000.))
    (_, _, z) = limb.coords_from_vector3(Vector3((0., 6500., 0.)), obs=OBSERVER, axes=3)

    assert not z.mask


def test_the_normal_points_away_from_the_body() -> None:
    """The normal at a limb point is the outward direction from the ground point."""

    normal = _limb().normal(Vector3((0., 6500., 0.)))

    assert normal.unit().vals == pytest.approx([0., 1., 0.], abs=1.e-9)


def test_intercept_lies_on_the_line_of_sight() -> None:
    """The intercept satisfies intercept = obs + t * los."""

    limb = _limb()
    los = Vector3((-1., 0.02, 0.))
    (pos, t) = limb.intercept(OBSERVER, los)

    assert pos.vals == pytest.approx((OBSERVER + los * t).vals, rel=1.e-9)


def test_limb_survives_a_pickle_round_trip() -> None:
    """Pickling restores the ground surface and the limits."""

    limb = _limb(limits=(0., 1000.))
    restored = pickle.loads(pickle.dumps(limb))

    assert isinstance(restored, Limb)
    assert restored.limits == limb.limits
    assert restored.ground.origin == limb.ground.origin

##########################################################################################
