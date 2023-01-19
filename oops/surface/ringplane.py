################################################################################
# oops/surface/ringplane.py: RingPlane subclass of class Surface
################################################################################

import numpy as np
from polymath     import Scalar, Vector3
from oops.frame   import Frame
from oops.path    import Path
from oops.surface import Surface

class RingPlane(Surface):
    """A subclass of Surface describing a flat surface in the (x,y) plane, in
    which the optional velocity field is defined by circular Keplerian motion
    about the center point. Coordinates are cylindrical (radius, longitude,
    elevation), with an optional offset in elevation from the equatorial (z=0)
    plane.

    Optional modes can be used to apply sinusoidal offset patterns in the radial
    coordinate.
    """

    COORDINATE_TYPE = 'polar'
    IS_VIRTUAL = False

    #===========================================================================
    def __init__(self, origin, frame, radii=None, gravity=None,
                       elevation=0., modes=[], epoch=0.):
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

            modes       an optional list of zero or more radial modes in the
                        ring. Each mode is described by a tuple of four
                        parameters (cycles, amp, peri0, speed):
                            cycles  the number of radial cycles around the ring.
                            amp     radial amplitude in km.
                            peri0   longitude of one radial minimum at epoch, in
                                    radians.
                            speed   the pattern speed in radians per second.

            epoch       the epoch at which the radial mode parameters apply.
                        Not used unless radial modes are present.
         """

        self.origin    = Path.as_waypoint(origin)
        self.frame     = Frame.as_wayframe(frame)
        self.gravity   = gravity
        self.elevation = float(elevation)
        self.modes     = modes
        self.nmodes    = len(self.modes)
        self.epoch     = float(epoch)

        if radii is None:
            self.radii = None
        else:
            self.radii    = np.asfarray(radii)
            self.radii_sq = self.radii**2

        # Save the unmasked version of this surface
        if radii is None:
            self.unmasked = self
        else:
            self.unmasked = RingPlane(self.origin, self.frame,
                                      radii = None,
                                      gravity = self.gravity,
                                      elevation = self.elevation,
                                      modes = self.modes,
                                      epoch = self.epoch)

        # Unique key for intercept calculations
        # ('ring', origin, frame, elevation, i, node, dnode_dt, epoch)
        # Extra elements are so OrbitPlane and RingPlane can share the same
        # key in situations where the orbit is not inclined.
        self.intercept_key = ('ring', self.origin.waypoint,
                                      self.frame.wayframe,
                                      self.elevation, 0., 0., 0., 0.)

    def __getstate__(self):
        return (Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame),
                None if self.radii is None else tuple(self.radii),
                self.gravity, self.elevation, self.modes, self.epoch)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=0., axes=2,
                                  derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for
                        solid surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        if axes not in (2, 3):
            raise ValueError('Surface.coords_from_vector3 ' +
                             'axes values must equal 2 or 3')

        pos = Vector3.as_vector3(pos, derivs)

        # Generate cylindrical coordinates
        (x,y,z) = pos.to_scalars()
        r = (x**2 + y**2).sqrt()
        theta = y.arctan2(x) % Scalar.TWOPI

        if self.nmodes:
            a = r - self.mode_offset(theta, time, derivs)
        else:
            a = r

        # Apply mask as needed
        if self.radii is not None:
            mask = a.tvl_lt(self.radii[0]) | a.tvl_gt(self.radii[1])
            if mask.any():
                a = a.remask_or(mask.vals)
                theta = theta.remask(a.mask)
                if axes > 2:
                    z = z.remask(r.mask)

        if axes == 2:
            return (a, theta)
        elif self.elevation == 0:
            return (a, theta, z)
        else:
            return (a, theta, z - self.elevation)

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=0, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        if len(coords) not in (2, 3):
            raise ValueError('Surface.vector3_from_coords requires 2 or 3 '
                             'coords')

        a = Scalar.as_scalar(coords[0], derivs)
        theta = Scalar.as_scalar(coords[1], derivs)

        if self.nmodes:
            r = a + self.mode_offset(theta, time, derivs)
        else:
            r = a

        if len(coords) > 2:
            z = Scalar.as_scalar(coords[2] + self.elevation, derivs)
        else:
            z = Scalar.as_scalar(self.elevation, derivs)

        x = r * theta.cos()
        y = r * theta.sin()

        return Vector3.from_scalars(x, y, z)

    #===========================================================================
    def intercept(self, obs, los, time=0., derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            time        a Scalar time at which to evaluate the surface.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
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
            if np.any(mask):
                pos = pos.remask_or(mask)
            t = t.remask(pos.mask)

        return (pos, t)

    #===========================================================================
    def normal(self, pos, time=0., derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface.
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
            if np.any(mask):
                perp = perp.remask_or(mask)

        return perp

    #===========================================================================
    def velocity(self, pos, time=0.):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface.

        Return:         a Vector3 of velocities, in units of km/s.
        """

        pos = Vector3.as_vector3(pos, False)

        # Handle special case that's easy
        if self.gravity is None and self.nmodes == 0:
            return Vector3(np.zeros(pos.vals.shape), pos.mask)

        # Generate info about intercept points
        (x,y,z) = pos.to_scalars(recursive=False)
        radius = (x**2 + y**2).sqrt()

        r_vector = Vector3.from_scalars(x,y,0.)

        # Handle radial modes
        if self.nmodes > 0:
            lon = y.arctan2(x)
            a = radius - self.mode_offset(lon, time)
            dr_dt = self.dmode_dt(lon, time)
            v_radial = dr_dt * (r_vector / radius)
        else:
            a = radius
            v_radial = None

        # Calculate the velocity field
        if self.gravity is None:
            v_angular = None
        else:
            n = Scalar(self.gravity.n(radius.values))
            v_angular = Vector3.ZAXIS.cross(r_vector) * n

        # Sum
        if v_radial is None:
            vflat = v_angular
        elif v_angular is None:
            vflat = v_radial
        else:
            vflat = v_radial + v_angular

        # The velocity is undefined outside the ring's radial limits
        if self.radii is not None:
            if self.radii[0] == 0:  # Avoids a hole in the middle due to modes
                mask = (a > self.radii[1])
            else:
                mask = (a < self.radii[0]) | (a > self.radii[1])

            if np.any(mask):
                vflat = vflat.remask_or(mask)

        return vflat

    ############################################################################
    # Radius conversions
    ############################################################################

    def mode_offset(self, lon, time, derivs=False):
        """Sum of the modes as a local radial offset."""

        offset = 0.
        for mode in self.modes:
            (cycles, amp, peri0, speed) = mode
            arg = Scalar(cycles * (lon - peri0) + (speed * (time - self.epoch)))
            offset = offset + amp * arg.cos(recursive=derivs)

        return offset

    #===========================================================================
    def dmode_dt(self, lon, time):
        """Sum of the radial velocities associated with the modes."""

        dr_dt = 0.
        for mode in self.modes:
            (cycles, amp, peri0, speed) = mode
            arg = Scalar(cycles * (lon - peri0) + (speed * (time - self.epoch)))

#           offset = offset + amp * arg.cos(), reversed
            dr_dt = dr_dt + (amp * speed) * arg.sin()

        return dr_dt

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.constants import TWOPI

class Test_RingPlane(unittest.TestCase):

    def runTest(self):

        from oops.gravity import Gravity
        from oops.event import Event

        np.random.seed(8829)

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

        ########################################################################
        # Test of radial modes
        ########################################################################

        # Coordinate/vector conversions
        refplane = RingPlane(Path.SSB, Frame.J2000)

        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 0.)], epoch=0.)

        obs = 10.e3 * np.random.rand(2,4,3,3)

        (a,theta,z) = plane.coords_from_vector3(obs, time=0., axes=3)
        test = plane.vector3_from_coords((a,theta,z), time=0.)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        test = plane.vector3_from_coords((a,theta,z), time=1.e8)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 2*np.pi/100.)], epoch=0.)

        obs = 10.e3 * np.random.rand(2,4,3,3)

        (a,theta,z) = plane.coords_from_vector3(obs, time=0., axes=3)
        test = plane.vector3_from_coords((a,theta,z), time=0.)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        test = plane.vector3_from_coords((a,theta,z), time=100.)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        # longitudes are the same in both maps
        (a0,theta0,z0) = refplane.coords_from_vector3(obs, time=0., axes=3)
        self.assertEqual(theta0, theta)

        # radial offsets are out of phase when time=50.
        diff1 = a - a0
        (a,theta,z) = plane.coords_from_vector3(obs, time=50., axes=3)
        diff2 = a - a0
        self.assertTrue(abs(diff1 + diff2).max() < 1.e-11)

        ########################################################################
        # Test of velocities
        ########################################################################

        pos = 10.e3 * np.random.rand(200,3)
        pos[...,2] = 0.     # set Z-coordinate to zero

        # No gravity, no modes
        refplane = RingPlane(Path.SSB, Frame.J2000)

        vels = refplane.velocity(obs)
        self.assertEqual(vels, (0.,0.,0.))

        # No gravity, motionless mode
        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 0.)], epoch=0.)

        vels = plane.velocity(obs)
        self.assertEqual(vels, (0.,0.,0.))

        # No gravity, modes
        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 2.*np.pi/100.)], epoch=0.)

        TIME = 90.
        (a0,theta0) = plane.coords_from_vector3(pos, time=TIME - 0.5)
        (a ,theta ) = plane.coords_from_vector3(pos, time=TIME)
        (a1,theta1) = plane.coords_from_vector3(pos, time=TIME + 0.5)
        self.assertEqual(theta, theta0)
        self.assertEqual(theta, theta1)

        vels = plane.velocity(pos, time=TIME)
        sep = vels.sep(pos)
        test = (sep + np.pi/2) % np.pi - np.pi/2
        self.assertTrue(abs(test).max() < 1.e-15)

        sign = 1 - 2 * (sep / np.pi)
        speed1 = sign * vels.norm()
        speed2 = a1 - a0
        diff = (speed2 - speed1) / abs(speed1).max()
        self.assertTrue(abs(diff).max() < 3.e-4)

        # Gravity, no modes
        plane = RingPlane(Path.SSB, Frame.J2000, gravity=Gravity.SATURN)

        (a, theta) = plane.coords_from_vector3(pos)

        vels = plane.velocity(pos)
        sep = vels.sep(pos)
        self.assertTrue(abs(sep - np.pi/2.).max() < 1.e-14)

        speed1 = vels.norm()
        speed2 = a * Gravity.SATURN.n(a.vals)
        diff = (speed2 - speed1) / speed1
        self.assertTrue(abs(diff).max() < 1.e-15)

        ########################################################################
        # coords_of_event, event_from_coords
        ########################################################################

        plane = RingPlane(Path.SSB, Frame.J2000)

        pos = Vector3(np.random.rand(2,4,3,3))
        vel = Vector3(np.random.rand(2,4,3,3))
        pos.insert_deriv('t', vel)

        event = Event(0., pos, Path.SSB, Frame.J2000)
        coords = plane.coords_of_event(event)
        test = plane.event_at_coords(0., coords)

        self.assertTrue(np.all(np.abs(test.pos.vals - pos.vals) < 1.e-15))
        self.assertTrue(np.all(np.abs(test.vel.vals - vel.vals) < 1.e-15))

        ########################################################################
        # Note: Additional unit testing is performed in orbitplane.py
        ########################################################################

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
