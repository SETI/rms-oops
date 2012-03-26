################################################################################
# oops_/surface/ansa.py: Ansa subclass of class Surface
#
# 2/27/12 Checked in (BSW)
# 3/24/12 MRS - revised for new surface API.
################################################################################

import numpy as np

from oops_.surface.surface_ import Surface
from oops_.array.all import *
import oops_.registry as registry

class Ansa(Surface):
    """The Ansa surface is defined as the locus of points where a radius vector
    from the pole of the Z-axis is perpendicular to the line of sight. This 
    provides a convenient coordinate system for describing rings when viewed
    nearly edge-on. The coordinates are (r,z,theta) where
        r       radial distance from the Z-axis, positive on the "right" side
                (if Z is pointing "up"); negative on the left side.
        z       vertical distance from the (x,y) plane.
        theta   angular distance from the ansa, with positive values further
                away from the observer and negative values closer.
    """

    COORDINATE_TYPE = "cylindrical"

    def __init__(self, origin, frame):
        """Constructor for an Ansa Surface.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring system.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z == 0).
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

    def coords_from_vector3(self, pos, obs, axes=2, derivs=False):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         position of the observer in the surface frame.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        pos and obs are returned as well. Using a tuple, you
                        can indicate whether to return partial derivatives on a
                        coordinate-by-coordinate basis.

        Return:         coordinate values packaged as a tuple containing two
                        unitless Scalars, one for each coordinate (r,z,theta).

                        Where derivs is True, the Scalar returned will have
                        subfields "d_dpos" and "d_obs", which contain the
                        partial derivatives of that coordinate, represented as
                        MatrixN objects with item shape [1,3].
        """

        pos = Vector3.as_standard(pos)
        obs = Vector3.as_standard(obs)
        (pos_x, pos_y, pos_z) = pos.as_scalars()
        (obs_x, obs_y, obs_z) = obs.as_scalars()

        rabs   = (pos_x**2 + pos_y**2).sqrt()
        obs_xy = (obs_x**2 + obs_y**2).sqrt()

        # Find the longitude of pos relative to obs
        lon = pos_y.arctan2(pos_x) - obs_y.arctan2(obs_x)

        # Put it in the range -pi to pi
        lon = ((lon + np.pi) % (2*np.pi)) - np.pi
        sign = lon.sign()

        r = rabs * sign
        z = pos_z.copy()

        # Fill in the third coordinate if necessary
        if axes > 2:
            # As discussed in the math found below with vector3_from_coords(),
            # the ansa longitude relative to the observer is:

            phi = (rabs / obs_xy).arccos()
            theta = sign*lon - phi
            coords = (r, z, theta)

        else:
            coords = (r, z)

        # Check the derivative requirements
        if np.any(derivs):
            if derivs is True: derivs = (derivs, derivs, derivs)

            if derivs[0]:
                dr_dpos = pos/r
                dr_dpos.vals[...,2] = 0.
                r.insert_subfield("d_dpos", dr_dpos.as_row())
                r.insert_subfield("d_dobs", Vector3((0,0,0)).as_row())

            if derivs[1]:
                z.insert_subfield("d_dpos",Vector3((0,0,1)).as_row())
                z.insert_subfield("d_dobs",Vector3((0,0,0)).as_row())

            if axes > 2 and derivs[2]:
                raise NotImplementedError("Ansa.coords_from_vector3() " +
                                    "does not implement theta derivatives")

        return coords

    def vector3_from_coords(self, coords, obs, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars: (r, z, theta)
                r       radial distance from the planetary pole.
                z       vertical distance from the equatorial plane.
                theta   longitude relative to the ansa.
            obs         position of the observer in the surface frame.
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

        # Given (r,z, theta) and the observer position, solve for position.
        #   pos = (|r| cos(a), |r| sin(a), z)
        # where angle a is defined by the location of the observer.
        #
        # theta = 0 at the ansa, where los and pos are perpendicular.
        # theta < 0 for points closer along the los, > 0 for points further.
        #
        # First solve for a where theta = 0.
        #
        #   pos_xy dot (obs_xy - pos_xy) = 0
        #   pos_xy dot pos_xy = pos_xy dot obs_xy
        #   r**2 = |r| cos(a) obs_x + |r| sin(a) obs_y
        #
        # For convenience, define the coordinate system so that obs falls on the
        # (x,z) plane, so obs_y = 0 and obs_x > 0.
        #
        #   r**2 = |r| obs_x cos(a)
        #
        #   cos(a) = |r| / obs_x
        #
        #   a = sign * arccos(|r| / obs_x)
        #
        # Define phi as the arccos term:
        #
        #   a = sign * phi(r,obs_x)
        #
        # Two solutions exist, symmetric about the (x,z) plane, as expected. The
        # positive sign corresponds to ring longitudes ahead of the observer,
        # which we define as the "right" ansa. The negative sign identifies the
        # "left" ansa.
        #
        # Theta is an angular offset from phi, with smaller values closer to the
        # observer and larger angles further away.

        assert len(coords) in {2,3}

        r = Scalar.as_standard(coords[0])
        z = Scalar.as_standard(coords[1])

        sign = r.sign()
        rabs = r * sign

        if len(coords) == 2:
            theta = Scalar(0.)
        else:
            theta = Scalar.as_standard(coords[2])

        (obs_x, obs_y, obs_z) = Vector3.as_standard(obs).as_scalars()
        obs_xy = (obs_x**2 + obs_y**2).sqrt()

        phi = (rabs / obs_xy).arccos()

        pos_lon = obs_y.arctan2(obs_x) + sign * (phi + theta)

        pos = Vector3.from_scalars(rabs * pos_lon.cos(),
                                   rabs * pos_lon.sin(), z)

        if derivs:
            raise NotImplementedError("Ansa.vector3_from_coords() " +
                                      "does not implement derivatives")

        return pos

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

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

        # (obs_xy + t los_xy) dot los_xy = 0
        # t = -(obs_xy dot los_xy) / (los_xy dot los_xy)
        # pos = obs + t * los

        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        obs_x = obs.as_scalar(0).vals
        obs_y = obs.as_scalar(1).vals
        los_x = los.as_scalar(0).vals
        los_y = los.as_scalar(1).vals

        los_x_sq = los_x**2
        los_y_sq = los_y**2
        los_sq = los_x_sq + los_y_sq

        obs_dot_los = obs_x * los_x + obs_y * los_y
        t_vals = -obs_dot_los / los_sq

        pos = obs + los * t_vals
        t = Scalar(t_vals, pos.mask)

        if derivs:
            # t = -(obs_x * los_x + obs_y * los_y) / |los|**2
            #
            # dt/dobs[x] = -los[x] / |los|**2
            # dt/dobs[y] = -los[y] / |los|**2
            # dt/dobs[z] = 0.
            #
            # dpos[x]/dobs[x] = 1 + los[x] dt/dobs[x]
            #                 = 1 - los[x]**2 / |los|**2
            #                 = (|los|**2 - los[x]**2) / |los|**2
            #                 = los[y]**2 / |los|**2
            # dpos[x]/dobs[y] = los[x] * dt/dobs[y]
            #                 = -los[x] * los[y] / |los|**2
            # dpos[x]/dobs[z] = los[x] * dt/dobs[z] = 0.
            #
            # dpos[z]/dobs[x] = los[z] * dt/dobs[x]
            #                 = -los[x] los[z] / |los|**2
            # dpos[z]/dobs[z] = 1

            los_z = los.as_scalar(2).vals

            dt_dobs_vals = np.zeros(obs.shape + [1,3])
            dt_dobs_vals[...,0,0] = -los_x / los_sq
            dt_dobs_vals[...,0,1] = -los_y / los_sq

            dpos_dobs_vals = np.zeros(obs.shape + [3,3])
            dpos_dobs_vals[...,0,0] = los_y_sq
            dpos_dobs_vals[...,0,1] = -los_x * los_y
            dpos_dobs_vals[...,1,0] = dpos_dobs_vals[...,0,1]
            dpos_dobs_vals[...,1,1] = los_x_sq
            dpos_dobs_vals[...,2,0] = -los_x * los_z
            dpos_dobs_vals[...,2,1] = -los_y * los_z
            dpos_dobs_vals[...,2,2] = los_sq
            dpos_dobs_vals /= los_sq[..., np.newaxis, np.newaxis]

            # t = -(obs_x * los_x + obs_y * los_y) / (los_x**2 + los_y**2)
            #
            # dt/dlos[x] = (-obs_x los_sq + 2 los_x * obs_dot_los) / los_sq**2
            # dt/dlos[y] = (-obs_y los_sq + 2 los_y * obs_dot_los) / los_sq**2
            # dt/dlos[z] = 0.
            #
            # dpos[x]/dlos[x] = los[x] dt/dlos[x] + t
            # dpos[x]/dlos[y] = los[x] dt/dlos[y]
            # dpos[y]/dlos[x] = los[y] dt/dlos[x]
            # dpos[y]/dlos[y] = los[y] dt/dlos[y] + t
            # dpos[z]/dlos[x] = los[z] dt/dlos[x]
            # dpos[z]/dlos[y] = los[z] dt/dlos[y]
            # dpos[z]/dlos[z] = t

            dt_dlos_vals = np.zeros(obs.shape + [1,3])
            dt_dlos_vals[...,0,0] = 2 * los_x * obs_dot_los - obs_x * los_sq
            dt_dlos_vals[...,0,1] = 2 * los_y * obs_dot_los - obs_y * los_sq
            dt_dlos_vals /= (los_sq**2)[...,np.newaxis,np.newaxis]

            dpos_dlos_vals = np.zeros(obs.shape + [3,3])
            dpos_dlos_vals[...,0,0] = los_x * dt_dlos_vals[...,0,0] + t_vals
            dpos_dlos_vals[...,0,1] = los_x * dt_dlos_vals[...,0,1]
            dpos_dlos_vals[...,1,0] = los_y * dt_dlos_vals[...,0,0]
            dpos_dlos_vals[...,1,1] = los_y * dt_dlos_vals[...,0,1] + t_vals
            dpos_dlos_vals[...,2,0] = los_z * dt_dlos_vals[...,0,0]
            dpos_dlos_vals[...,2,1] = los_z * dt_dlos_vals[...,0,1]
            dpos_dlos_vals[...,2,2] = t_vals

            # Normalize in the los denominators
            los_norm_vals = los.norm().vals[..., np.newaxis, np.newaxis]
            dt_dlos_vals *= los_norm_vals
            dpos_dlos_vals *= los_norm_vals

            pos.d_dobs = MatrixN(dpos_dobs_vals, pos.mask)
            pos.d_dlos = MatrixN(dpos_dlos_vals, pos.mask)
            t.d_dobs = MatrixN(dt_dobs_vals, pos.mask)
            t.d_dlos = MatrixN(dt_dlos_vals, pos.mask)

        return (pos, t)

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface.
        Counterintuitively, we define this as the ring plane normal for the
        Ansa surface, so that incidence and emission angles work out as the
        user would expect.

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

        perp = Vector3((0,0,1))

        if derivs:
            perp.insert_subfield("d_dpos", MatrixN(np.zeros((3,3))))

        return perp

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Ansa(unittest.TestCase):
    
    def runTest(self):

        import oops_.frame.all
        import oops_.path.all

        surface = Ansa("SSB", "J2000")

        # intercept()
        obs = Vector3(np.random.rand(100,3) * 1.e5)
        los = Vector3(np.random.rand(100,3))

        (pos,t) = surface.intercept(obs, los)
        pos_xy = pos * (1,1,0)
        los_xy = los * (1,1,0)

        self.assertTrue(abs(pos_xy.sep(los_xy) - np.pi/2) < 1.e-8)
        self.assertTrue(abs(obs + t * los - pos) < 1.e-8)

        # coords_from_vector3()
        obs = Vector3(np.random.rand(100,3) * 1.e6)
        pos = Vector3(np.random.rand(100,3) * 1.e5)

        (r,z) = surface.coords_from_vector3(pos, obs, axes=2)

        pos_xy = pos * (1,1,0)
        pos_z  = pos.as_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)) < 1.e-8)
        self.assertTrue(abs(pos_z - z) < 1.e-8)

        (r,z,theta) = surface.coords_from_vector3(pos, obs, axes=3)

        pos_xy = pos * (1,1,0)
        pos_z  = pos.as_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)) < 1.e-8)
        self.assertTrue(abs(pos_z - z) < 1.e-8)
        self.assertTrue(abs(theta) <= np.pi)

        # vector3_from_coords()
        obs = Vector3(1.e-5 + np.random.rand(100,3) * 1.e6)
        r = Scalar(1.e-4 + np.random.rand(100) * 9e-4)
        z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
        theta = Scalar(np.random.rand(100))

        pos = surface.vector3_from_coords((r,z), obs)

        pos_xy = pos * (1,1,0)
        pos_z  = pos.as_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)) < 1.e-8)
        self.assertTrue(abs(pos_z - z) < 1.e-8)

        obs_xy = obs * (1,1,0)
        self.assertTrue(abs(pos_xy.sep(obs_xy - pos_xy) - np.pi/2) < 1.e-5)

        pos1 = surface.vector3_from_coords((r,z,theta), obs)
        pos1_xy = pos1 * (1,1,0)
        self.assertTrue(pos1_xy.sep(pos_xy) - theta < 1.e-5)

        pos1 = surface.vector3_from_coords((r,z,-theta), obs)
        pos1_xy = pos1 * (1,1,0)
        self.assertTrue(pos1_xy.sep(pos_xy) - theta < 1.e-5)

        pos = surface.vector3_from_coords((-r,z), obs)
        pos_xy = pos * (1,1,0)

        pos1 = surface.vector3_from_coords((-r,z,-theta), obs)
        pos1_xy = pos1 * (1,1,0)
        self.assertTrue(pos1_xy.sep(pos_xy) - theta < 1.e-5)

        pos1 = surface.vector3_from_coords((-r,z,theta), obs)
        pos1_xy = pos1 * (1,1,0)
        self.assertTrue(pos1_xy.sep(pos_xy) - theta < 1.e-5)

        # vector3_from_coords() & coords_from_vector3()
        obs = Vector3((1.e6,0,0))
        r = Scalar(1.e4 + np.random.rand(100) * 9.e4)
        r *= np.sign(2 * np.random.rand(100) - 1)
        z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
        theta = Scalar((2 * np.random.rand(100) - 1) * 1.)

        pos = surface.vector3_from_coords((r,z,theta), obs)
        coords = surface.coords_from_vector3(pos, obs, axes=3)
        self.assertTrue(abs(r - coords[0]) < 1.e-5)
        self.assertTrue(abs(z - coords[1]) < 1.e-5)
        self.assertTrue(abs(theta - coords[2]) < 1.e-8)

        obs = Vector3(np.random.rand(100,3) * 1.e6)
        pos = Vector3(np.random.rand(100,3) * 1.e5)
        coords = surface.coords_from_vector3(pos, obs, axes=3)
        test_pos = surface.vector3_from_coords(coords, obs)
        self.assertTrue(abs(test_pos - pos) < 1.e-5)

        # intercept() derivatives
        obs = Vector3(np.random.rand(100,3))
        los = Vector3(np.random.rand(100,3))
        (pos0,t0) = surface.intercept(obs, los, derivs=True)

        eps = 1e-6
        (pos1,t1) = surface.intercept(obs + (eps,0,0), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,0]) < 1.e-6)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,0,0]) < 1.e-6)

        (pos1,t1) = surface.intercept(obs + (0,eps,0), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,1]) < 1.e-5)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,0,1]) < 1.e-6)

        (pos1,t1) = surface.intercept(obs + (0,0,eps), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,2]) < 1.e-5)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,0,2]) < 1.e-6)

        eps = 1e-6
        los_norm = los.norm()

        (pos1,t1) = surface.intercept(obs, los + (eps,0,0), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps * los_norm
        dt_dlos_test = (t1 - t0) / eps * los_norm
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,0]) < 3.e-3)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,0,0]) < 3.e-3)

        (pos1,t1) = surface.intercept(obs, los + (0,eps,0), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps * los_norm
        dt_dlos_test = (t1 - t0) / eps * los_norm
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,1]) < 3.e-3)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,0,1]) < 3.e-3)

        (pos1,t1) = surface.intercept(obs, los + (0,0,eps), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps * los_norm
        dt_dlos_test = (t1 - t0) / eps * los_norm
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,2]) < 3.e-3)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,0,2]) < 3.e-3)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
