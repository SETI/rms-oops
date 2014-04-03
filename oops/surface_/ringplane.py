################################################################################
# oops/surface_/ringplane.py: RingPlane subclass of class Surface
################################################################################

import numpy as np

from polymath import *
import gravity

from oops.surface_.surface import Surface
from oops.path_.path       import Path
from oops.frame_.frame     import Frame
from oops.constants        import *

class RingPlane(Surface):
    """RingPlane is a subclass of Surface describing a flat surface in the (x,y)
    plane, in which the optional velocity field is defined by circular Keplerian
    motion about the center point. Coordinate are cylindrical (radius,
    longitude, elevation), with an optional offset in elevation from the
    equatorial (z=0) plane."""

    COORDINATE_TYPE = "polar"
    IS_VIRTUAL = False

    def __init__(self, origin, frame, radii=None, gravity=None,
                       elevation=0.):
        """Constructor for a RingPlane surface.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z = 0).

            radii       the nominal inner and outer radii of the ring, in km.
                        None for a ring with no radial limits.

            gravity     an optional Gravity object, used to define the orbital
                        velocities within the plane.

            elevation   an optional offset of the ring plane in the direction of
                        positive rotation, in km.
            """

        self.origin    = Path.as_waypoint(origin)
        self.frame     = Frame.as_wayframe(frame)
        self.gravity   = gravity
        self.elevation = elevation

        if radii is None:
            self.radii = None
        else:
            self.radii    = np.asfarray(radii)
            self.radii_sq = self.radii**2

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer observer positions. Ignored for
                        solid surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.
        """

        pos = Vector3.as_vector3(pos, derivs)

        # Generate cylindrical coordinates
        (x,y,z) = pos.to_scalars()
        r = (x**2 + y**2).sqrt()
        theta = y.arctan2(x) % TWOPI

        if axes == 2:
            return (r, theta)
        elif self.elevation == 0:
            return (r, theta, z)
        else:
            return (r, theta, z - self.elevation)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        r = Scalar.as_scalar(coords[0], derivs)
        theta = Scalar.as_scalar(coords[1], derivs)

        if len(coords) > 2:
            z = Scalar.as_scalar(coords[2] + self.elevation, derivs)
        else:
            z = Scalar.as_scalar(self.elevation)

        x = r * theta.cos()
        y = r * theta.sin()

        return Vector3.from_scalars(x, y, z)

    def intercept(self, obs, los, derivs=False, t_guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            t_guess     initial guess at the t array, optional.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a unitless Scalar such that:
                            position = obs + t * los
        """

        # Solve for obs + factor * los for scalar t, such that the z-component
        # equals zero.
        obs = Vector3.as_vector3(obs, derivs)
        los = Vector3.as_vector3(los, derivs)

        obs_z = obs.to_scalar(2)
        los_z = los.to_scalar(2)

        t = (self.elevation - obs_z)/los_z
        pos = obs + t * los

        # Mask based on radial limits if necessary
        if self.radii is not None:
            r_sq = pos.norm_sq(False)
            mask = (r_sq < self.radii_sq[0]) | (r_sq > self.radii_sq[1])
            pos = pos.mask_where(mask)
            t = t.mask_where(mask)

        return (pos, t)

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        pos = Vector3.as_vector3(pos, derivs)

        # Always the Z-axis
        perp = pos.all_constant((0.,0.,1.))

        # The normal is undefined outside the ring's radial limits
        if self.radii is not None:
            r_sq = pos.norm_sq(False)
            mask = (r_sq < self.radii_sq[0]) | (r_sq > self.radii_sq[1])
            perp = perp.mask_where(mask)

        return perp

    def velocity(self, pos):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        pos = Vector3.as_vector3(pos, False)

        # Calculate the velocity field
        if self.gravity is None:
            return Vector3(np.zeros(pos.vals.shape), pos.mask)

        radius = pos.norm()
        n = Scalar(self.gravity.n(radius.values))
        vflat = Vector3.ZAXIS.cross(pos) * n

        # The velocity is undefined outside the ring's radial limits
        if self.radii is not None:
            mask = (radius < self.radii[0]) | (radius > self.radii[1])
            vflat = vflat.mask_where(mask)

        return vflat

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RingPlane(unittest.TestCase):

    def runTest(self):

        plane = RingPlane(Path.SSB, Frame.J2000)

        # Coordinate/vector conversions
        obs = np.random.rand(2,4,3,3)

        (r,theta,z) = plane.coords_from_vector3(obs,axes=3)
        self.assertTrue((theta >= 0.).all())
        self.assertTrue((theta < TWOPI).all())
        self.assertTrue((r >= 0.).all())

        test = plane.vector3_from_coords((r,theta,z))
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-15))

        # Ring intercepts
        los = np.random.rand(2,4,3,3)
        obs[...,2] =  np.abs(obs[...,2])
        los[...,2] = -np.abs(los[...,2])

        (pts, factors) = plane.intercept(obs, los)
        self.assertTrue(abs(pts.to_scalar(2)).max() < 1.e-15)

        angles = pts - obs
        self.assertTrue((angles.sep(los) > -1.e-12).all())
        self.assertTrue((angles.sep(los) <  1.e-12).all())

        # Intercepts that point away from the ring plane
        self.assertTrue(np.all(factors.vals > 0.))

        # Note: Additional unit testing is performed in orbitplane.py

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
