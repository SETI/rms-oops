##########################################################################################
# tests/surface/test_polarlimb.py
##########################################################################################

import numpy as np
import pytest

from polymath               import Vector3
from oops.constants         import HALFPI
from oops.surface.polarlimb import PolarLimb
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.spheroid  import Spheroid
from oops.surface.ellipsoid import Ellipsoid


def test_polarlimb():

    REQ  = 60268.
    RMID = 54364.
    RPOL = 50000.

    ground = Spheroid('SSB', 'J2000', (REQ, RPOL))
    limb = PolarLimb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[...,0] = -4 *REQ
    los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

    perp = limb.normal(track)
    assert abs(perp.sep(los) - HALFPI).max() < 1.e-12

    coords = limb.coords_from_vector3(cept, obs=obs, axes=3)
    assert abs(coords[2]).max() < 1.e6

    cept2 = limb.vector3_from_coords(coords, obs=obs)
    assert (cept2 - cept).norm().median() < 1.e-10

    ######################################################################################

    ground = Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
    limb = PolarLimb(ground)

    obs = Vector3([4*REQ,0,0])

    los_vals = np.empty((220,220,3))
    los_vals[...,0] = -4 *REQ
    los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
    los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
    los = Vector3(los_vals)

    (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

    perp = limb.normal(track)
    assert abs(perp.sep(los) - HALFPI).max() < 1.e-12

    coords = limb.coords_from_vector3(cept, obs=obs, axes=3)
    assert abs(coords[2]).max() < 1.e6

    cept2 = limb.vector3_from_coords(coords, obs=obs)
    assert (cept2 - cept).norm().median() < 1.e-10

    Path._reset_caches()
    Frame._reset_caches()

##########################################################################################
# Hints, groundtracks, and the third coordinate
##########################################################################################

REQ = 60268.
RPOL = 50000.

# An observer four equatorial radii away along the x-axis, and lines of sight that graze
# the limb well above the surface
OBS = Vector3([4 * REQ, 0., 0.])
LOS = Vector3([(-4. * REQ, 0.6 * REQ, 0.4 * REQ), (-4. * REQ, -0.7 * REQ, 0.3 * REQ)])


def _limb() -> PolarLimb:
    """A polar limb around an oblate spheroid.

    Returns:
        PolarLimb: The limb surface.
    """

    return PolarLimb(Spheroid('SSB', 'J2000', (REQ, RPOL)))


def test_the_hint_returned_by_one_call_serves_the_next() -> None:
    """The coefficient p is handed back so a second call can start from it."""

    limb = _limb()
    (cept, _, _) = limb.intercept(OBS, LOS, hints=True)

    (z, clock, p) = limb.coords_from_vector3(cept, obs=OBS, hints=True)
    (z2, clock2, p2) = limb.coords_from_vector3(cept, obs=OBS, hints=p)

    assert z2.vals == pytest.approx(z.vals, abs=1.e-6)
    assert clock2.vals == pytest.approx(clock.vals, abs=1.e-9)
    assert p2 == p


def test_the_groundtrack_lies_on_the_body_surface() -> None:
    """The groundtrack is the point on the spheroid below the limb point."""

    limb = _limb()
    (cept, _, _) = limb.intercept(OBS, LOS, hints=True)

    (z, _, track) = limb.coords_from_vector3(cept, obs=OBS, axes=2, groundtrack=True)[0:3]

    assert limb._ground.coords_from_vector3(track, axes=3)[2].vals \
           == pytest.approx(0., abs=1.e-6)

    # The elevation is the distance from the groundtrack, signed by which side it is on
    assert (cept - track).norm().vals == pytest.approx(np.abs(z.vals), abs=1.e-6)


def test_the_third_coordinate_measures_the_offset_along_the_line_of_sight() -> None:
    """A point displaced along the line of sight has that displacement as its distance."""

    limb = _limb()
    (cept, _, _) = limb.intercept(OBS, LOS, hints=True)
    los = cept - OBS
    moved = cept + 1000. * los.unit()

    (_, _, d) = limb.coords_from_vector3(moved, obs=OBS, axes=3)

    assert d.vals == pytest.approx(1000., abs=1.e-6)


def test_the_coordinates_invert_with_the_offset_included() -> None:
    """The three coordinates map back to the point they were derived from."""

    limb = _limb()
    (cept, _, _) = limb.intercept(OBS, LOS, hints=True)
    moved = cept + 1000. * (cept - OBS).unit()
    coords = limb.coords_from_vector3(moved, obs=OBS, axes=3)

    back = limb.vector3_from_coords(coords, obs=OBS)

    assert (back - moved).norm().max() < 1.e-6


def test_vector3_from_coords_hands_back_the_hints_and_the_groundtrack() -> None:
    """The hint given is returned unchanged, and the groundtrack is on the surface."""

    limb = _limb()
    coords = limb.coords_from_vector3(limb.intercept(OBS, LOS)[0], obs=OBS)

    (pos, hints, track) = limb.vector3_from_coords(coords, obs=OBS, hints='reused',
                                                   groundtrack=True)

    assert hints == 'reused'
    assert limb._ground.coords_from_vector3(track, axes=3)[2].vals \
           == pytest.approx(0., abs=1.e-6)
    assert (pos - limb.intercept(OBS, LOS)[0]).norm().max() < 1.e-6

##########################################################################################
