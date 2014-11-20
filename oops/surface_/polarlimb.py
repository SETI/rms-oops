################################################################################
# oops/surface_/polarlimb.py: PolarLimb subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface   import Surface
from oops.surface_.limb      import Limb

from oops.config             import SURFACE_PHOTONS, LOGGING
from oops.constants          import *

class PolarLimb(Surface):
    """The PolarLimb surface is a variant of the Limb surface, in which the
    coordinates are redefined as follows:
        z       the elevation above the surface, the vertical distance from the
                surface.
        clock   an angle on the sky, measured clockwise from the projected
                direction of the north pole.
        d       offset distance beyond the virtual limb plane along the line of
                sight.
    """

    COORDINATE_TYPE = "polar"
    IS_VIRTUAL = True
    DEBUG = False   # True for convergence testing in intercept()

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = False
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = False
    intercept_DERIVS_ARE_IMPLEMENTED = False
    normal_DERIVS_ARE_IMPLEMENTED = False

    def __init__(self, ground, limits=None):
        """Constructor for a Limb surface.

        Input:
            ground      the Surface object relative to which limb points are to
                        be defined. It should be a Spheroid or Ellipsoid,
                        optically using Centric or Graphic coordinates.
            limits      an optional single value or tuple defining the absolute
                        numerical limit(s) placed on z; values outside this
                        range are masked.
        """

        assert ground.COORDINATE_TYPE == "spherical"
        self.ground = ground
        self.origin = ground.origin
        self.frame  = ground.frame

        self.limb = Limb(self.ground, limits)

        if limits is None:
            self.limits = None
        else:
            self.limits = (limits[0], limits[1])

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False,
                                  groundtrack=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            groundtrack True to add an extra returned value to the tuple,
                        containing the groundtrack point as a Vector3.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.

                        if groundtrack is True, a Vector3 of ground points is
                        appended to the returned tuple.
        """

        if derivs:
            raise NotImplementedError("PolarLimb.coords_from_vector3() " +
                                      "does not implement derivatives")

        pos = Vector3.as_vector3(pos, False)
        obs = Vector3.as_vector3(obs, False)
        los = pos - obs
        (cept, t, track) = self.limb.intercept(obs, los, groundtrack=True)

        (z, clock) = self.limb.z_clock_from_intercept(cept, obs)

        if axes == 2:
            results = (z, clock)
        else:
            d = los.dot(pos - cept) / los.norm()
            results = (z, clock, d)

        if groundtrack:
            results += (track,)

        return results

    def vector3_from_coords(self, coords, obs=None, derivs=False,
                                  groundtrack=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
            obs         position of the observer in the surface frame.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.
            groundtrack True to replace the returned value by a tuple, where the
                        second quantity is the groundtrack point as a Vector3.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.
        """

        if derivs:
            raise NotImplementedError("PolarLimb.vector3_from_coords() " +
                                      "does not implement derivatives")

        (cept, track) = self.limb.intercept_from_z_clock(z, clock, obs,
                                                         derivs=False,
                                                         groundtrack=True)

        if len(coords) > 2:
            d = Scalar.as_scalar(coords[2], False)
            los = cept - obs
            cept += (d / los.norm()) * los

        if groundtrack:
            return (cept, track)
        else:
            return cept

    def intercept(self, obs, los, derivs=False, guess=None, groundtrack=False):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los
            groundtrack True to include the surface intercept points of the body
                        associated with each limb intercept. This array can
                        speed up any subsequent calculations such as calls to
                        normal(), and can be used to determine locations in
                        body coordinates.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        if derivs:
            raise NotImplementedError("PolarLimb.intercept() " +
                                      "does not implement derivatives")

        return self.limb.intercept(obs, los, derivs, guess, groundtrack)

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        return self.ground.normal(pos, derivs)

################################################################################
# (lon,lat) conversions
################################################################################

    def lonlat_from_vector3(pos, derivs=False, groundtrack=True):
        """Longitude and latitude for a position near the surface."""

        track = self.ground.intercept_normal_to(pos, derivs=derivs)
        coords = self.ground.coords_from_vector3(track, derivs=derivs)

        if groundtrack:
            return (coords[0], coords[1], track)
        else:
            return coords[:2]

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Limb(unittest.TestCase):

    def runTest(self):

        from oops.frame_.frame import Frame
        from oops.path_.path import Path

        from oops.surface_.spheroid  import Spheroid
        from oops.surface_.ellipsoid import Ellipsoid

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.

        NPTS = 1000

        ground = Spheroid("SSB", "J2000", (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

        perp = limb.normal(track)
        self.assertTrue(abs(perp.sep(los) - HALFPI).max() < 1.e-12)

        coords = limb.coords_from_vector3(cept, obs, axes=3)
        self.assertTrue(abs(coords[2]).max() < 1.e6)

        cept2 = limb.vector3_from_coords(coords, obs)
        self.assertTrue((cept2 - cept).norm().median() < 1.e-10)

        ####################

        ground = Ellipsoid("SSB", "J2000", (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

        perp = limb.normal(track)
        self.assertTrue(abs(perp.sep(los) - HALFPI).max() < 1.e-12)

        coords = limb.coords_from_vector3(cept, obs, axes=3)
        self.assertTrue(abs(coords[2]).max() < 1.e6)

        cept2 = limb.vector3_from_coords(coords, obs)
        self.assertTrue((cept2 - cept).norm().median() < 1.e-10)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
