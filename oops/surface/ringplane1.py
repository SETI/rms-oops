import numpy as np
import unittest
import gravity

import oops

############################################
# RingPlane
############################################

class RingPlane(oops.Surface):
    """A flat surface in the (x,y) plane, in which the points are optionally
    undergoing circular Keplerian motion about the center point."""

    def __init__(self, origin, frame, gravity=None):

        """Constructor for a RingPlane object.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z = 0).

            gravity     an optional Gravity object, used to define the orbital
                        velocities within the plane.
            """

        self.origin_id = oops.as_path_id(origin)
        self.frame_id  = oops.as_frame_id(frame)
        self.gravity = gravity

    def as_coords(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            position        a Vector3 of positions at or near the surface.
            axes            2 or 3, indicating whether to return a tuple of two
                            or 3 Scalar objects.

        Return:             coordinate values packaged as a tuple containing
                            two or three Scalars, one for each coordinate.
         """

        position = oops.Vector3.as_vector3(position)
        x = position.vals[...,0]
        y = position.vals[...,1]

        r     = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x) % (2.*np.pi)

        if axes > 2:
            z = position.vals[...,2]
            return (oops.Scalar(r), oops.Scalar(theta), oops.Scalar(z))
        else:
            return (oops.Scalar(r), oops.Scalar(theta))

    def as_vector3(self, r, theta, z=oops.Scalar(0.)):
        """Converts coordinates in the surface's internal coordinate system into
        position vectors at or near the surface.

        Input:
            r               Scalar of radius values.
            theta           Scalar of longitude values.
            z               Optional Scalar of elevation values.

        Return:             the corresponding Vector3 of positions.
        """

        # Convert to Scalars
        r     = oops.Scalar(r)
        theta = oops.Scalar(theta)
        z     = oops.Scalar(z)

        # Convert to vectors
        shape = oops.Array.broadcast_shape((r, theta, z), [3])
        array = np.empty(shape)
        array[...,0] = np.cos(theta.vals) * r.vals
        array[...,1] = np.sin(theta.vals) * r.vals
        array[...,2] = z.vals

        return oops.Vector3(array)

    def intercept(self, obs, los):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs             observer position as a Vector3.
            los             line of sight as a Vector3.

        Return:             a tuple (position, factor)
            position        a Vector3 of intercept points on the surface.
            factor          a Scalar of factors such that
                                position = obs + factor * los
        """

        # Solve for obs + t * los for scalar t, such that the z-component
        # equals zero.
        obs = oops.Vector3.as_vector3(obs).vals
        los = oops.Vector3.as_vector3(los).vals

        t = (obs[...,2]/los[...,2])[..., np.newaxis]
        array = obs - t * los

        # Make the z-component exact
        array[...,2] = 0.

        return (oops.Vector3(array), oops.Scalar(-t[...,0]))

    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.

        Input:
            position        a Vector3 of positions at or near the surface.

        Return:             a Vector3 containing directions normal to the
                            surface that pass through the position. Lengths are
                            arbitrary.
        """

        return oops.Vector3((0,0,1))# This does not inherit the given vector's
                                    # shape, but should broadcast properly

    def gradient_at_position(self, position, axis=0, projected=True):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient is defined as the vector pointing in the direction
        of most rapid change in the value of a particular surface coordinate.

        The magnitude of the gradient vector is the rate of change of the
        coordinate value when starting from this point and moving in this
        direction.

        Input:
            position        a Vector3 of positions at or near the surface.

            axis            0, 1 or 2, identifying the coordinate axis for which
                            the gradient is sought.

            projected       True to project the gradient into the surface if
                            necessary. This has no effect on a RingPlane.
        """

        if axis == 3: return oops.Vector3([0,0,1])

        radii = position.copy()
        radii.vals[...,2] = 0.

        if axis == 0:
            return radii.unit()

        if axis == 1:
            vectors = oops.Vector3(np.zeros(position.vals.shape))
            vectors.vals[...,0] =  position.vals[...,1]
            vectors.vals[...,1] = -position.vals[...,0]

            return vectors / radii.norm()**2

        raise ValueError("illegal axis value")

    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            position        a Vector3 of positions at or near the surface.
        """

        if self.gravity == None: return oops.Vector3((0,0,0))

        # TBD: We need a vector form of the gravity library!

        position = oops.Vector3.as_vector3(position)
        radius = oops.Vector3(position).norm()
        # n = gravity.n(radius.vals)        # !!! Not yet implemented
        n = 0.

        return oops.Vector3(position.cross((0,0,-1)) * n)

    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal          a Vector3 of normal vectors.

        Return:             a Vector3 of surface intercept points. Where no
                            solution exists, the components of the returned
                            vector should be np.nan.
        """

        pass

        # For a flat ring plane this is a degenerate problem. The function
        # returns (0,0,0) as the location where every exactly perpendicular
        # vector intercepts the plane. It returns 3-vectors of np.nan everywhere
        # else.
        buffer = np.zeros(normal.shape + [3])
        buffer[normal.vals[...,0] != 0., 0:3] = np.nan
        buffer[normal.vals[...,1] != 0., 0:3] = np.nan

        return oops.Vector3(buffer)

    def intercept_normal_to(self, position):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            position        a Vector3 of positions near the surface.

        Return:             a Vector3 of surface intercept points. Where no
                            solution exists, the components of the returned
                            vector should be np.nan.
        """

        intercept = oops.Vector3.as_vector3(position).copy()
        intercept.vals[...,2] = 0.

        return intercept

########################################
# UNIT TESTS
########################################

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

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
