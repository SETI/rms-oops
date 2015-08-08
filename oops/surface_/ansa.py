################################################################################
# oops/surface_/ansa.py: Ansa subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface   import Surface
from oops.surface_.ringplane import RingPlane
from oops.path_.path         import Path
from oops.frame_.frame       import Frame
from oops.constants import *

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
    IS_VIRTUAL = True

    def __init__(self, origin, frame, gravity=None, ringplane=None):
        """Constructor for an Ansa Surface.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring system.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z == 0).

            gravity     an optional Gravity object, used to define the orbital
                        velocities relative to the surface. Also used if there
                        is a need to define an associated RingPlane frame.

            ringplane   used by static method for_ringplane(); otherwise it
                        should be ignored.
        """

        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame)
        self.gravity = gravity

        if ringplane is None:
            self.ringplane = RingPlane(self.origin, self.frame,
                                       gravity=self.gravity)
        else:
            self.ringplane = ringplane

    @staticmethod
    def for_ringplane(ringplane):
        """Construct an Ansa Surface associated with a given RingPlane.

        Input:
            ringplane   a ringplane surface relative to which this ansa surface
                        is to be defined.
        """

        return Ansa(ringplane.origin, ringplane.frame, ringplane.gravity,
                    ringplane)

    def coords_from_vector3(self, pos, obs, axes=2, derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        pos = Vector3.as_vector3(pos, derivs)
        obs = Vector3.as_vector3(obs, derivs)
        (pos_x, pos_y, pos_z) = pos.to_scalars()
        (obs_x, obs_y, obs_z) = obs.to_scalars()

        rabs   = (pos_x**2 + pos_y**2).sqrt()
        obs_xy = (obs_x**2 + obs_y**2).sqrt()

        # Find the longitude of pos relative to obs
        lon = pos_y.arctan2(pos_x) - obs_y.arctan2(obs_x)

        # Put it in the range -pi to pi
        lon = ((lon + PI) % TWOPI) - PI
        sign = lon.sign()
        r = rabs * sign

        # Fill in the third coordinate if necessary
        if axes > 2:
            # As discussed in the math found below with vector3_from_coords(),
            # the ansa longitude relative to the observer is:

            phi = (rabs / obs_xy).arccos()
            theta = sign*lon - phi
            return (r, pos_z, theta)

        return (r, pos_z)

    def vector3_from_coords(self, coords, obs, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
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

        r = Scalar.as_scalar(coords[0], derivs)
        z = Scalar.as_scalar(coords[1], derivs)

        sign = r.sign()
        rabs = r * sign

        if len(coords) == 2:
            theta = Scalar(0.)
        else:
            theta = Scalar.as_scalar(coords[2], derivs)

        (obs_x, obs_y, obs_z) = Vector3.as_vector(obs, derivs).to_scalars()
        obs_xy = (obs_x**2 + obs_y**2).sqrt()

        phi = (rabs / obs_xy).arccos()

        pos_lon = obs_y.arctan2(obs_x) + sign * (phi + theta)

        pos = Vector3.from_scalars(rabs * pos_lon.cos(),
                                   rabs * pos_lon.sin(), z)
        return pos

    def intercept(self, obs, los, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       Unused.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        obs = Vector3.as_vector3(obs, derivs)
        los = Vector3.as_vector3(los, derivs)

        # (obs_xy + t los_xy) dot los_xy = 0
        # t = -(obs_xy dot los_xy) / (los_xy dot los_xy)
        # pos = obs + t * los

        obs_x = obs.to_scalar(0)
        obs_y = obs.to_scalar(1)
        los_x = los.to_scalar(0)
        los_y = los.to_scalar(1)

        los_sq = los_x**2 + los_y**2

        obs_dot_los = obs_x * los_x + obs_y * los_y
        t = -obs_dot_los / los_sq

        pos = obs + t * los

        return (pos, t)

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.

        NOTE: Counterintuitively, we define this as the ring plane normal for
        the Ansa surface, so that incidence and emission angles work out as the
        user would expect.
        """

        pos = Vector3.as_vector3(pos, derivs)

        # Always the Z-axis
        return pos.all_constant((0.,0.,1.))

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Ansa(unittest.TestCase):
    
    def runTest(self):

        surface = Ansa("SSB", "J2000")

        # intercept()
        obs = Vector3( np.random.rand(10,3) * 1.e5)
        los = Vector3(-np.random.rand(10,3))

        (pos,t) = surface.intercept(obs, los)
        pos_xy = pos.element_mul((1,1,0))
        los_xy = los.element_mul((1,1,0))

        self.assertTrue(abs(pos_xy.sep(los_xy) - HALFPI).max() < 1.e-8)
        self.assertTrue(abs(obs + t * los - pos).max() < 1.e-8)

        # coords_from_vector3()
        obs = Vector3(np.random.rand(100,3) * 1.e6)
        pos = Vector3(np.random.rand(100,3) * 1.e5)

        (r,z) = surface.coords_from_vector3(pos, obs, axes=2)

        pos_xy = pos.element_mul(Vector3((1,1,0)))
        pos_z  = pos.to_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)).max() < 1.e-8)
        self.assertTrue(abs(pos_z - z).max() < 1.e-8)

        (r,z,theta) = surface.coords_from_vector3(pos, obs, axes=3)

        pos_xy = pos.element_mul(Vector3((1,1,0)))
        pos_z  = pos.to_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)).max() < 1.e-8)
        self.assertTrue(abs(pos_z - z).max() < 1.e-8)
        self.assertTrue(abs(theta).max() <= PI)

        # vector3_from_coords()
        obs = Vector3(1.e-5 + np.random.rand(100,3) * 1.e6)
        r = Scalar(1.e-4 + np.random.rand(100) * 9e-4)
        z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
        theta = Scalar(np.random.rand(100))

        pos = surface.vector3_from_coords((r,z), obs)

        pos_xy = pos.element_mul(Vector3((1,1,0)))
        pos_z  = pos.to_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)).max() < 1.e-8)
        self.assertTrue(abs(pos_z - z).max() < 1.e-8)

        obs_xy = obs.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos_xy.sep(obs_xy - pos_xy) - HALFPI).max() < 1.e-5)

        pos1 = surface.vector3_from_coords((r,z,theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        pos1 = surface.vector3_from_coords((r,z,-theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        pos = surface.vector3_from_coords((-r,z), obs)
        pos_xy = pos.element_mul(Vector3((1,1,0)))

        pos1 = surface.vector3_from_coords((-r,z,-theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        pos1 = surface.vector3_from_coords((-r,z,theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        # vector3_from_coords() & coords_from_vector3()
        obs = Vector3((1.e6,0,0))
        r = Scalar(1.e4 + np.random.rand(100) * 9.e4)
        r *= np.sign(2 * np.random.rand(100) - 1)
        z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
        theta = Scalar((2 * np.random.rand(100) - 1) * 1.)

        pos = surface.vector3_from_coords((r,z,theta), obs)
        coords = surface.coords_from_vector3(pos, obs, axes=3)
        self.assertTrue(abs(r - coords[0]).max() < 1.e-5)
        self.assertTrue(abs(z - coords[1]).max() < 1.e-5)
        self.assertTrue(abs(theta - coords[2]).max() < 1.e-8)

        obs = Vector3(np.random.rand(100,3) * 1.e6)
        pos = Vector3(np.random.rand(100,3) * 1.e5)
        coords = surface.coords_from_vector3(pos, obs, axes=3)
        test_pos = surface.vector3_from_coords(coords, obs)
        self.assertTrue(abs(test_pos - pos).max() < 1.e-5)

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
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,0]).max() < 1.e-6)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,0]).max() < 1.e-6)

        (pos1,t1) = surface.intercept(obs + (0,eps,0), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,1]).max() < 1.e-5)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,1]).max() < 1.e-6)

        (pos1,t1) = surface.intercept(obs + (0,0,eps), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,2]).max() < 1.e-5)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,2]).max() < 1.e-6)

        eps = 1e-6
        (pos1,t1) = surface.intercept(obs, los + (eps,0,0), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps
        dt_dlos_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,0]).max() < 1.e-2)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,0]).max() < 1.e-2)

        (pos1,t1) = surface.intercept(obs, los + (0,eps,0), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps
        dt_dlos_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,1]).max() < 1.e-2)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,1]).max() < 1.e-2)

        (pos1,t1) = surface.intercept(obs, los + (0,0,eps), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps
        dt_dlos_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,2]).max() < 1.e-2)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,2]).max() < 1.e-2)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
