##########################################################################################
# tests/surface/test_orbitplane.py
##########################################################################################

import numpy as np
import pytest

from polymath                import Scalar, Vector3
from oops.constants          import PI, HALFPI, TWOPI, RPD
from oops.event              import Event
from oops.path               import Path
from oops.surface.orbitplane import OrbitPlane


def test_orbitplane():
    # elements = (a, lon, n)

    # Circular orbit, no derivatives, forward
    elements = (1, 0, 1)
    epoch = 0
    orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000', path_id='TEST')

    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
    (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)

    r_true = Scalar([1,2,1,1])
    l_true = Scalar([0, 0, PI, HALFPI])
    z_true = Scalar([0,0,0,0.1])

    assert abs(r - r_true).max() < 1.e-12
    assert abs(l - l_true).max() < 1.e-12
    assert abs(z - z_true).max() < 1.e-12

    # Circular orbit, no derivatives, reverse
    pos2 = orbit.vector3_from_coords((r, l, z), derivs=False)

    assert (pos - pos2).norm().max() < 1.e-10

    # Circular orbit, with derivatives, forward
    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
    pos.insert_deriv('pos', Vector3.IDENTITY, override=True)
    eps = 1.e-6
    delta = 1.e-4

    for step in ([eps,0,0], [0,eps,0], [0,0,eps]):
        dpos = Vector3(step)
        (r,l,z) = orbit.coords_from_vector3(pos + dpos, axes=3,
                                            derivs=True)

        r_test = r + r.d_dpos.chain(dpos)
        l_test = l + l.d_dpos.chain(dpos)
        z_test = z + z.d_dpos.chain(dpos)

        assert abs(r - r_test).max() < delta
        assert abs(l - l_test).max() < delta
        assert abs(z - z_test).max() < delta

    # Circular orbit, with derivatives, reverse
    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
    (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
    eps = 1.e-6
    delta = 1.e-5

    r.insert_deriv('r', Scalar.ONE, override=True)
    l.insert_deriv('l', Scalar.ONE, override=True)
    z.insert_deriv('z', Scalar.ONE, override=True)
    pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

    pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dr
    assert (pos1_test - pos1).norm().max() < delta

    pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dl
    assert (pos1_test - pos1).norm().max() < delta

    pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dz
    assert (pos1_test - pos1).norm().max() < delta

    # elements = (a, lon, n, e, peri, prec)

    # Eccentric orbit, no derivatives, forward
    ae = 0.1
    prec = 0.1
    elements = (1, 0, 1, ae, 0, prec)
    epoch = 0
    orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000', path_id='TEST')
    eps = 1.e-6
    delta = 1.e-5

    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
    event = Event(0., pos, 'SSB', 'J2000')
    (r,l,z) = orbit.coords_of_event(event, derivs=False)

    r_true = Scalar([1. + ae, 2. + ae, 1 - ae, np.sqrt(1. + ae**2)])
    l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,ae)])
    z_true = Scalar([0,0,0,0.1])

    assert abs(r - r_true).max() < delta
    assert abs(l - l_true).max() < delta
    assert abs(z - z_true).max() < delta

    # Eccentric orbit, no derivatives, reverse
    event2 = orbit.event_at_coords(event.time, (r,l,z)).wrt_ssb()
    assert (pos - event2.pos).norm().max() < 1.e-10
    assert (event2.vel).norm().max() < 1.e-10

    # Eccentric orbit, with derivatives, forward
    ae = 0.1
    prec = 0.1
    elements = (1, 0, 1, ae, 0, prec)
    epoch = 0
    orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000')
    eps = 1.e-6
    delta = 3.e-5

    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])

    for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
        vel = Vector3(v)
        event = Event(0., (pos, vel), 'SSB', 'J2000')
        (r,l,z) = orbit.coords_of_event(event, derivs=True)

        event = Event(eps, (pos + vel*eps, vel), 'SSB', 'J2000')
        (r1,l1,z1) = orbit.coords_of_event(event, derivs=False)
        dr_dt_test = (r1 - r) / eps
        dl_dt_test = (l1 - l) / eps
        dz_dt_test = (z1 - z) / eps

        assert abs(r.d_dt - dr_dt_test).max() < delta
        assert abs(z.d_dt - dz_dt_test).max() < delta

        d_dl_dt = ((l.d_dt*eps - dl_dt_test*eps + PI) % TWOPI - PI) / eps
        assert abs(d_dl_dt).max() < delta

    # Eccentric orbit, with derivatives, reverse
    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
    (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
    eps = 1.e-6
    delta = 1.e-5

    r.insert_deriv('r', Scalar.ONE)
    l.insert_deriv('l', Scalar.ONE)
    z.insert_deriv('z', Scalar.ONE)
    pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

    pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dr
    assert (pos1_test - pos1).norm().max() < delta

    pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dl
    assert (pos1_test - pos1).norm().max() < delta

    pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dz
    assert (pos1_test - pos1).norm().max() < delta

    # elements = (a, lon, n, e, peri, prec, i, node, regr)

    # Inclined orbit, no eccentricity, no derivatives, forward
    inc = 0.1
    regr = -0.1
    node = -HALFPI
    sini = np.sin(inc)
    cosi = np.cos(inc)

    elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
    epoch = 0
    orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000')
    eps = 1.e-6
    delta = 1.e-5

    dz = 0.1
    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])
    event = Event(0., pos, 'SSB', 'J2000')
    (r,l,z) = orbit.coords_of_event(event, derivs=False)

    r_true = Scalar([cosi, 2*cosi, cosi, np.sqrt(1 + (dz*sini)**2)])
    l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,dz*sini)])
    z_true = Scalar([-sini, -2*sini, sini, dz*cosi])

    assert abs(r - r_true).max() < delta
    assert abs(l - l_true).max() < delta
    assert abs(z - z_true).max() < delta

    # Inclined orbit, no derivatives, reverse
    event2 = orbit.event_at_coords(event.time, (r,l,z)).wrt_ssb()
    assert (pos - event2.pos).norm().max() < 1.e-10
    assert event2.vel.norm().max() < 1.e-10

    # Inclined orbit, with derivatives, forward
    inc = 0.1
    regr = -0.1
    node = -HALFPI
    sini = np.sin(inc)
    cosi = np.cos(inc)

    elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
    epoch = 0
    orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000')
    eps = 1.e-6
    delta = 1.e-5

    dz = 0.1
    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])

    for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
        vel = Vector3(v)
        event = Event(0., (pos, vel), 'SSB', 'J2000')
        (r,l,z) = orbit.coords_of_event(event, derivs=True)

        event = Event(eps, (pos + vel*eps, vel), 'SSB', 'J2000')
        (r1,l1,z1) = orbit.coords_of_event(event, derivs=False)
        dr_dt_test = (r1 - r) / eps
        dl_dt_test = ((l1 - l + PI) % TWOPI - PI) / eps
        dz_dt_test = (z1 - z) / eps

        assert abs(r.d_dt - dr_dt_test).max() < delta
        assert abs(l.d_dt - dl_dt_test).max() < delta
        assert abs(z.d_dt - dz_dt_test).max() < delta

    # Inclined orbit, with derivatives, reverse
    pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
    (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
    eps = 1.e-6
    delta = 1.e-5

    r.insert_deriv('r', Scalar.ONE)
    l.insert_deriv('l', Scalar.ONE)
    z.insert_deriv('z', Scalar.ONE)
    pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

    pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dr
    assert (pos1_test - pos1).norm().max() < delta

    pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dl
    assert (pos1_test - pos1).norm().max() < delta

    pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
    pos1_test = pos0 + eps * pos0.d_dz
    assert (pos1_test - pos1).norm().max() < delta

    # From/to mean anomaly
    elements = (1, 0, 1, 0.1, 0, 0.1)
    epoch = 0
    orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000', path_id='TEST')

    l = np.arange(361) * RPD
    anoms = orbit.to_mean_anomaly(l)

    lons = orbit.from_mean_anomaly(anoms)
    assert abs(lons - l).max() < 1.e-15


# A representative eccentric orbit: 100,000 km, with a mean motion and pericenter fixed in
# inertial space so that the surface frame does not rotate and the orbital velocity can be
# compared directly against the Keplerian solution.
_A = 100000.
_N = 1.e-4
_ANOMALIES = np.arange(24) * (TWOPI / 24.)


def _eccentric_orbit(e):
    """An OrbitPlane of eccentricity `e` with its pericenter along the x-axis.

    Parameters:
        e (float): The orbital eccentricity.

    Returns:
        OrbitPlane: The orbit, centered on the SSB and defined in the J2000 frame.
    """

    return OrbitPlane((_A, 0., _N, e, 0., 0.), 0., 'SSB', 'J2000')


def _planet_offset(orbit):
    """The position of the planet relative to the center of the eccentric ring.

    The surface is centered on the displaced ring center, so this is what converts a
    planet-centered position into one relative to the surface. It is obtained from the
    orbit's own origin path rather than assumed.

    Parameters:
        orbit (OrbitPlane): The orbit to evaluate.

    Returns:
        Vector3: The offset, in km, in the orbit's own frame.
    """

    ring_center = Path.as_path(orbit.origin).event_at_time(0.).wrt('SSB', orbit.frame)
    return -ring_center.pos


def _kepler_velocity_error(orbit, e):
    """The largest relative error in `velocity` against the Keplerian solution.

    The comparison is made at points spread around the orbit. Because the surface models
    the orbit only to first order in eccentricity, the error is expected to be of order
    e**2.

    Parameters:
        orbit (OrbitPlane): The orbit to evaluate.
        e (float): The orbital eccentricity.

    Returns:
        float: The largest error, relative to the local orbital speed.
    """

    offset = _planet_offset(orbit)
    worst = 0.
    for nu in _ANOMALIES:
        # The exact Keplerian position and velocity, relative to the planet
        r = _A * (1. - e**2) / (1. + e * np.cos(nu))
        v_radial = _N * _A / np.sqrt(1. - e**2) * e * np.sin(nu)
        v_tangential = _N * _A / np.sqrt(1. - e**2) * (1. + e * np.cos(nu))

        pos = Vector3((r * np.cos(nu), r * np.sin(nu), 0.)) + offset
        (vx, vy, _) = [value.vals for value in orbit.velocity(pos).to_scalars()]

        radial = vx * np.cos(nu) + vy * np.sin(nu)
        tangential = -vx * np.sin(nu) + vy * np.cos(nu)

        speed = np.hypot(v_radial, v_tangential)
        error = np.hypot(radial - v_radial, tangential - v_tangential) / speed
        worst = max(worst, error)

    return worst


def _kepler_orbit(e):
    """An OrbitPlane with the given eccentricity, for the anomaly conversions.

    Parameters:
        e (float): The orbital eccentricity.

    Returns:
        OrbitPlane: The orbit, unregistered.
    """

    return OrbitPlane((_A, 0., _N, e, 0., 0.), 0., 'SSB', 'J2000')


def test_velocity_of_a_circular_orbit() -> None:
    """A circular orbit moves at the mean motion, perpendicular to its radius."""

    orbit = _eccentric_orbit(0.)
    pos = Vector3([(_A, 0., 0.), (0., _A, 0.), (-_A, 0., 0.), (0., 0.5*_A, 0.)])

    expected = _N * Vector3.ZAXIS.cross(pos)
    assert (orbit.velocity(pos) - expected).norm().max() < 1.e-15


def test_velocity_of_an_eccentric_orbit_matches_kepler() -> None:
    """An eccentric orbit moves at the Keplerian velocity, to first order in e."""

    e = 0.02
    assert _kepler_velocity_error(_eccentric_orbit(e), e) < 3. * e**2


def test_velocity_error_is_second_order_in_eccentricity() -> None:
    """The departure from the Keplerian velocity falls as the square of eccentricity.

    A first-order model leaves an error of order e**2, so halving the eccentricity
    quarters it. An error that instead falls only in proportion to e would mean a term of
    the wrong order, such as a displacement applied in the wrong direction.
    """

    errors = [_kepler_velocity_error(_eccentric_orbit(e), e) for e in (0.02, 0.01, 0.005)]

    for (coarse, fine) in zip(errors[:-1], errors[1:]):
        assert 3.5 < coarse / fine < 4.5


def test_to_mean_anomaly_inverts_from_mean_anomaly() -> None:
    """The two conversions are exact inverses for an eccentricity the model can solve."""

    lon = Scalar(np.arange(0., TWOPI, 0.01))

    for e in (0.001, 0.01, 0.1, 0.3, 0.45):
        orbit = _kepler_orbit(e)
        assert abs(orbit.from_mean_anomaly(orbit.to_mean_anomaly(lon))
                   - lon).max(builtins=True) < 1.e-14


def test_to_mean_anomaly_reports_a_failure_to_converge() -> None:
    """An eccentricity the iteration cannot solve raises rather than returning a guess.

    The derivative of the longitude with respect to the anomaly approaches zero as the
    eccentricity approaches 0.5, so Newton's method breaks down there. It used to stop at
    whatever it had reached and return it, a value that could be many radians wrong.
    """

    lon = Scalar(np.arange(0., TWOPI, 0.01))

    for e in (0.5, 0.8):
        with pytest.raises(ValueError, match='did not converge'):
            _kepler_orbit(e).to_mean_anomaly(lon)


def test_to_mean_anomaly_accepts_masked_longitudes() -> None:
    """A masked longitude has nothing to solve and must not be read as a failure."""

    orbit = _kepler_orbit(0.2)

    assert np.all(orbit.to_mean_anomaly(Scalar([1., 2., 3.], mask=True)).mask)

    partly = orbit.to_mean_anomaly(Scalar([1., 2., 3.], mask=[False, True, False]))
    assert list(partly.mask) == [False, True, False]

##########################################################################################
