################################################################################
# oops/surface/ringplane.py: RingPlane subclass of class Surface
#
# 2/8/12 Modified (MRS) - Updated for style; added elevation parameter; added
#   mask tracking.
################################################################################

import numpy as np
import gravity

from baseclass import Surface
from oops.xarray.all import *
import oops.frame.registry as frame_registry
import oops.path.registry as path_registry

class RingPlane(Surface):
    """RingPlane is a subclass of Surface describing a flat surface in the (x,y)
    plane, in which the optional velocity field is defined by circular Keplerian
    motion about the center point. Coordinate are cylindrical (radius,
    longitude, elevation), with an optional offset in elevation from the
    equatorial (z=0) plane."""

    def __init__(self, origin, frame, gravity=None, elevation=0.):
        """Constructor for a RingPlane object.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z = 0).

            gravity     an optional Gravity object, used to define the orbital
                        velocities within the plane.

            elevation   an optional offset of the ring plane in the direction of
                        positive rotation, in km.
            """

        self.origin_id = path_registry.as_id(origin)
        self.frame_id  = frame_registry.as_id(frame)
        self.gravity   = gravity
        self.elevation = elevation

    def as_coords(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.
        """

        # Convert to a Vector3 and strip units, if any
        position = Vector3.as_standard(position)
        mask = position.mask

        x = position.vals[...,0]
        y = position.vals[...,1]

        r     = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x) % (2.*np.pi)

        if axes > 2:
            z = position.vals[...,2] - self.elevation
            return (Scalar(r,mask), Scalar(theta,mask), Scalar(z,mask))
        else:
            return (Scalar(r,mask), Scalar(theta,mask))

    def as_vector3(self, r, theta, z=0.):
        """Converts coordinates in the surface's internal coordinate system into
        position vectors at or near the surface.

        Input:
            coord1      a Scalar of values for the first coordinate, with
                        optional units.
            coord2      a Scalar of values for the second coordinate, with
                        optional units.
            coord3      a Scalar of values for the third coordinate, with
                        optional units; default is Scalar(0.).

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:         the corresponding unitless Vector3 object of positions,
                        in km.
        """

        # Convert to Scalars and strip units, if any
        r     = Scalar.as_standard(r)
        theta = Scalar.as_standard(theta)
        z     = Scalar.as_standard(z)

        # Convert to vectors
        shape = Array.broadcast_shape((r, theta, z), [3])
        array = np.empty(shape)
        array[...,0] = np.cos(theta.vals) * r.vals
        array[...,1] = np.sin(theta.vals) * r.vals
        array[...,2] = z.vals + self.elevation

        return Vector3(array, r.mask | theta.mask | z.mask)

    def intercept(self, obs, los):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.

        Return:         a tuple (position, factor)
            position    a unitless Vector3 of intercept points on the surface,
                        in km.
            factor      a unitless Scalar of factors such that:
                            position = obs + factor * los
        """

        # Solve for obs + t * los for scalar t, such that the z-component
        # equals zero.
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        t = ((self.elevation - obs.vals[...,2]) / los.vals[...,2])
        array = obs.vals + t[..., np.newaxis] * los.vals

        # Make the z-component exact
        array[...,2] = self.elevation

        mask = obs.mask | los.mask | (los.vals[...,2] == 0) | (t < 0.)

        return (Vector3(array,mask), Scalar(t,mask))

    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.
        """

        return Vector3((0,0,1))     # This does not inherit the given vector's
                                    # shape, but should broadcast properly

    def gradient(self, position, axis=0, projected=True):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient is defined as the vector pointing in the direction
        of most rapid change in the value of a particular surface coordinate.

        The magnitude of the gradient vector is the rate of change of the
        coordinate value when starting from this point and moving in this
        direction.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.

            axis        0, 1 or 2, identifying the coordinate axis for which the
                        gradient is sought.

            projected   True to project the gradient vector into the surface.

        Return:         a unitless Vector3 of the gradients sought. Values are
                        always in standard units.
        """

        if axis == 3: return Vector3([0,0,1])

        position = Vector3.as_standard(position)

        radii = position.copy()
        radii.vals[...,2] = 0.

        if axis == 0:
            return radii.unit()

        if axis == 1:
            mask = position.mask
            vectors = Vector3(np.zeros(position.vals.shape), mask)
            vectors.vals[...,0] =  position.vals[...,1]
            vectors.vals[...,1] = -position.vals[...,0]

            return vectors / radii.norm()**2

        raise ValueError("illegal axis value")

    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        if self.gravity is None: return Vector3((0,0,0))

        position = Vector3.as_standard(position)
        radius = position.norm()
        n = self.gravity.n(radius.vals)

        return Vector3(position.cross((0,0,-1)) * n)

    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal      a Vector3 of normal vectors, with optional units.

        Return:         a unitless Vector3 of surface intercept points, in km.
                        Where no solution exists, the components of the returned
                        vector should be masked.
        """

        # For a flat ring plane this is a degenerate problem. The function
        # returns (0,0,0) as the location where every exactly perpendicular
        # vector intercepts the plane. It returns masked values everywhere
        # else.

        normal = Vector3.as_standard(normal)

        buffer = np.zeros(normal.shape + [3])
        mask = (normal.mask | normal.vals[...,0] != 0.
                            | normal.vals[...,1] != 0.)

        return Vector3(buffer, mask)

    def intercept_normal_to(self, position):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            position    a Vector3 of positions near the surface, with optional
                        units.

        Return:         a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.
        """

        intercept = Vector3.as_standard(position).copy()
        intercept.vals[...,2] = 0.

        return intercept

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RingPlane(unittest.TestCase):

    def runTest(self):

        plane = RingPlane("SSB", "J2000")

        # Coordinate/vector conversions
        obs = np.random.rand(2,4,3,3)

        (r,theta,z) = plane.as_coords(obs,axes=3)
        self.assertTrue(theta >= 0.)
        self.assertTrue(theta < 2.*np.pi)
        self.assertTrue(r >= 0.)

        test = plane.as_vector3(r,theta,z)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-15))

        # Ring intercepts
        los = np.random.rand(2,4,3,3)
        obs[...,2] =  np.abs(obs[...,2])
        los[...,2] = -np.abs(los[...,2])

        (pts, factors) = plane.intercept(obs, los)
        self.assertTrue(pts.as_scalar(2) == 0.)

        angles = pts - obs
        self.assertTrue(angles.sep(los) > -1.e-12)
        self.assertTrue(angles.sep(los) <  1.e-12)

        # Intercepts that point away from the ring plane
        self.assertTrue(np.all(factors.vals > 0.))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
