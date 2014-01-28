################################################################################
# oops/surface_/ringplane.py: RingPlane subclass of class Surface
#
# 2/8/12 Modified (MRS) - Updated for style; added elevation parameter; added
#   mask tracking.
# 2/17/12 Modified (MRS) - Added optional radial limits to the definition of a
#   RingPlane.
# 3/2/12 MRS: Completed implementation of all the derivatives; updated the
#   comments.
################################################################################

import numpy as np
import gravity

from oops.surface_.surface import Surface
from oops.array_ import *
import oops.registry as registry

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

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)
        self.gravity   = gravity
        self.elevation = elevation

        if radii is None:
            self.radii = None
        else:
            self.radii    = np.asfarray(radii)
            self.radii_sq = self.radii**2

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Converts from position vectors in the internal frame to the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on an coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        # Convert to a Vector3 and strip units, if any
        pos = Vector3.as_standard(pos)
        mask = pos.mask

        # Generate cylindrical coordinates
        x = pos.vals[...,0]
        y = pos.vals[...,1]

        r     = Scalar(np.sqrt(x**2 + y**2), mask)
        theta = Scalar(np.arctan2(y,x) % (2.*np.pi), mask)

        if axes > 2:
            z = Scalar(pos.vals[...,2] - self.elevation, mask)

        # Generate derivatives if necessary
        if np.any(derivs):
            if np.shape(derivs) == (): derivs = (derivs, derivs, derivs)

            if derivs[0]:
                dr_dpos = VectorN(pos.vals[...]*(1,1,0), pos.mask).unit()
                r.insert_subfield("d_dpos", dr_dpos.as_row())
                r.insert_subfield("d_dobs", MatrixN.ZERO3_ROW)

            if derivs[1]:
                dtheta_dpos = Vector3.ZAXIS.cross(pos) / r**2
                theta.insert_subfield("d_dpos", dtheta_dpos.as_row())
                theta.insert_subfield("d_dobs", MatrixN.ZERO3_ROW)

            if axes > 2 and derivs[2]:
                z.insert_subfield("d_dpos", MatrixN.ZAXIS_ROW)

        if axes > 2:
            return (r, theta, z)
        else:
            return (r, theta)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                r       a Scalar of radius values, with optional units.
                theta   a Scalar of longitude values, with optional units.
                z       an optional Scalar of elevation values, with optional
                        units; default is Scalar(0.).
            obs         position of the observer in the surface frame; ignored.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

                        If derivs is True, then pos is returned with subfields
                        "d_dobs" and "d_dcoords", where the former contains the
                        MatrixN of partial derivatives with respect to obs and
                        the latter is the MatrixN of partial derivatives with
                        respect to the coordinates. The MatrixN item shapes are
                        [3,3].
        """

        # Convert to Scalars and strip units, if any
        r     = Scalar.as_standard(coords[0])
        theta = Scalar.as_standard(coords[1])

        if len(coords) == 2:
            z = Scalar(0.)
        else:
            z = Scalar.as_standard(coords[2])

        # Convert to vectors
        shape = Array.broadcast_shape((r, theta, z))
        vals = np.empty(shape + [3])

        cos_theta = np.cos(theta.vals)
        sin_theta = np.sin(theta.vals)
        vals[...,0] = cos_theta * r.vals
        vals[...,1] = sin_theta * r.vals
        vals[...,2] = z.vals + self.elevation

        pos = Vector3(vals, r.mask | theta.mask | z.mask)

        # Generate derivatives if necessary
        if derivs:
            dpos_dcoord_vals = np.zeros(shape + [3,3])
            dpos_dcoord_vals[...,0,0] =  cos_theta
            dpos_dcoord_vals[...,1,0] =  sin_theta
            dpos_dcoord_vals[...,0,1] = -sin_theta * r.vals
            dpos_dcoord_vals[...,1,1] =  cos_theta * r.vals
            dpos_dcoord_vals[...,2,2] = 1.

            pos.insert_subfield("d_dcoord", MatrixN(dpos_dcoord_vals, pos.mask))

        return pos

    def intercept(self, obs, los, derivs=False, t_guess=None):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.
            t_guess     initial guess at the t array, optional.

        Return:         (pos, t)
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar of scale factors t such that:
                            position = obs + t * los

                        If derivs is True, then pos and t are returned with
                        subfields "d_dobs" and "d_dlos", where the former
                        contains the MatrixN of partial derivatives with respect
                        to obs and the latter is the MatrixN of partial
                        derivatives with respect to los. The MatrixN item shapes
                        are [3,3] for the derivatives of pos, and [1,3] for the
                        derivatives of t. For purposes of differentiation, los
                        is assumed to have unit length.
        """

        # Solve for obs + factor * los for scalar t, such that the z-component
        # equals zero.
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        obs_z = obs.as_scalar(axis=2)
        los_z = los.as_scalar(axis=2)

        t = (self.elevation - obs_z)/los_z
        pos = obs + t * los

        # Make the z-component exact
        pos.vals[...,2] = self.elevation

        # Mask based on radial limits if necessary
        if self.radii is not None:
            r_sq = pos.vals[...,0]**2 + pos.vals[...,1]**2
            pos.mask |= (r_sq < self.radii_sq[0]) | (r_sq > self.radii_sq[1])
            t.mask = pos.mask

        # Calculate derivatives if necessary
        if derivs:

            # Our equations to differentiate are...
            #
            #   t = (self.elevation - obs[z]) / los[z]
            #   pos = obs + t(obs[z],los[z]) * los
            #
            # Taking derivatives...
            #   dt/dobs[x] = 0
            #   dt/dobs[y] = 0
            #   dt/dobs[z] = -1 / los[z]
            #
            #   dt/dlos[x] = 0
            #   dt/dlos[y] = 0
            #   dt/dlos[z] = (self.elevation - obs[z]) * (-1) / los[z]**2
            #              = -t / los[z]
            #
            #   dpos[x]/dobs[x] = 1
            #   dpos[x]/dobs[z] = los[x] * dt/dobs[z]
            #   dpos[y]/dobs[y] = 1
            #   dpos[z]/dobs[z] = 1 + dt/dobs[z] * los[z]
            #                   = 1 + (-1 / los[z]) * los[z]
            #                   = 0 (which should have been obvious)
            #
            #   dpos[x]/dlos[x] = t
            #   dpos[x]/dlos[z] = los[x] * dt/dlos[z]
            #   dpos[y]/dlos[y] = t
            #   dpos[y]/dlos[z] = los[y] * dt/dlos[z]
            #   dpos[z]/dlos[z] = t + dt/dlos[z] * los[z]
            #                   = t + (-t / los[z]) * los[z]
            #                   = 0 (which also should have been obvious)

            dt_dobs_z = -1. / los_z

            vals = np.zeros(t.shape + [1,3])
            vals[...,0,2] = dt_dobs_z.vals

            dt_dobs = MatrixN(vals, t.mask)
            t.insert_subfield("d_dobs", dt_dobs)
            t.insert_subfield("d_dlos", dt_dobs * t)

            vals = np.zeros(t.shape + [3,3])
            vals[...,0,0] = 1.
            vals[...,0,2] = dt_dobs_z.vals * los.vals[...,0]
            vals[...,1,1] = 1.
            vals[...,1,2] = dt_dobs_z.vals * los.vals[...,1]

            dpos_dobs = MatrixN(vals, t.mask)
            pos.insert_subfield("d_dobs", dpos_dobs)
            pos.insert_subfield("d_dlos", dpos_dobs * t)

        return (pos, t)

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            derivs      True to include a matrix of partial derivatives.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.

                        If derivs is True, then the normal vectors returned have
                        a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        pos = Vector3.as_standard(pos)
        mask = pos.mask

        # The normal is undefined outside the ring's radial limits
        if self.radii is not None:
            r_sq = pos.vals[...,0]**2 + pos.vals[...,1]**2
            mask = (mask | (r_sq < self.radii_sq[0]) |
                           (r_sq > self.radii_sq[1]))

        vals = np.zeros(pos.vals.shape)
        vals[...,2] = 1.
        perp = Vector3(vals, mask)

        if derivs:
            perp.insert_subfield("d_dpos", MatrixN.ZERO33.copy())

        return perp

    def velocity(self, pos):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        pos = Vector3.as_standard(pos)

        # The velocity is undefined outside the ring's radial limits
        if self.radii is not None:
            r_sq = pos.vals[...,0]**2 + pos.vals[...,1]**2
            mask = (pos.mask | (r_sq < self.radii_sq[0]) |
                               (r_sq > self.radii_sq[1]))
        else:
            mask = pos.mask

        # Calculate the velocity field
        if self.gravity is None:
            return Vector3(np.zeros(pos.vals.shape), mask)

        radius = pos.norm()
        n = Scalar(self.gravity.n(radius.vals))

        vflat = pos.cross((0,0,-1)) * n
        return Vector3(vflat.vals, mask)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RingPlane(unittest.TestCase):

    def runTest(self):

        plane = RingPlane("SSB", "J2000")

        # Coordinate/vector conversions
        obs = np.random.rand(2,4,3,3)

        (r,theta,z) = plane.coords_from_vector3(obs,axes=3)
        self.assertTrue(theta >= 0.)
        self.assertTrue(theta < 2.*np.pi)
        self.assertTrue(r >= 0.)

        test = plane.vector3_from_coords((r,theta,z))
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

        # Note: Additional unit testing is performed in orbitplane.py

        registry.initialize()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
