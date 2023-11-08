################################################################################
# oops/surface/ringplane.py: RingPlane subclass of class Surface
################################################################################

import numpy as np
from polymath     import Scalar, Vector3
from oops.frame   import Frame
from oops.path    import Path
from oops.surface import Surface
from oops.gravity.oblategravity import OblateGravity

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

        # Identify the maximum orbital rate by any means necessary; without this
        # limit, speeds near the origin get ridiculous.
        if self.radii is not None:
            r = self.radii[0]
            self.max_rate = self.gravity.n(r)
        elif hasattr(self.gravity, 'rp'):
            r = self.gravity.rp
            self.max_rate = self.gravity.n(r)
        else:
            # If we can't figure out the planet, clamp the rate at that for an
            # orbit skimming the surface of Neptune. (Note that this rate is
            # faster than that for Jupiter, Saturn, or Uranus.)
            neptune = OblateGravity.NEPTUNE
            self.max_rate = neptune.n(neptune.rp)

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
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                       derivs=False, hints=None):
        """Surface coordinates associated with a position vector.

        Input:
            pos         a Vector3 of positions at or near the surface, relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        unless this RingPlane contains radial modes.
            axes        2 or 3, indicating whether to return the first two
                        coordinates (rad, theta) or all three (rad, theta, z) as
                        Scalars.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       ignored. Provided for compatibility with other Surface
                        subclasses.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
            rad         mean orbital radius in the ring plane, in km.
            theta       longitude in radians of the intercept point.
            z           vertical distance in km above the ring plane; included
                        if axes == 3.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)
        pos = Vector3.as_vector3(pos, derivs)

        # Generate cylindrical coordinates
        (r, theta, z) = pos.to_cylindrical()

        if self.modes:
            a = r - self._mode_offset(theta, time, derivs=derivs)
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
    def vector3_from_coords(self, coords, obs=None, time=0., derivs=False):
        """The position where a point with the given coordinates falls relative
        to this surface's origin and frame.

        Input:
            coords      a tuple of two or three Scalars defining coordinates at
                        or near this surface. These can have different shapes,
                        but must be broadcastable to a common shape.
                rad     mean orbital radius in the ring plane, in km.
                theta   longitude in radians of the intercept point.
                z       vertical distance in km above the ring plane.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        unless this RingPlane contains radial modes.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        a = Scalar.as_scalar(coords[0], derivs)
        theta = Scalar.as_scalar(coords[1], derivs)

        if self.modes:
            r = a + self._mode_offset(theta, time, derivs=derivs)
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
    def intercept(self, obs, los, time=None, direction='dep', derivs=False,
                                  guess=None, hints=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3 relative to this
                        surface's origin and frame.
            los         line of sight as a Vector3 in this surface's frame.
            time        a Scalar time at the surface; ignored here.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface; ignored here.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       unused.
            hints       if not None (the default), this value is appended to the
                        returned tuple. Needed for compatibility with other
                        Surface subclasses.

        Return:         a tuple (pos, t) or (pos, t, hints), where
            pos         a Vector3 of intercept points on the surface relative
                        to this surface's origin and frame, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
            hints       the input value of hints, included if hints is not None.
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

        if hints is not None:
            return (pos, t, hints)

        return (pos, t)

    #===========================================================================
    def normal(self, pos, time=0., derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        here.
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
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        unless this RingPlane contains radial modes.

        Return:         a Vector3 of velocities, in units of km/s.
        """

        pos = Vector3.as_vector3(pos, False)

        # Handle special case that's easy
        if self.gravity is None and not self.modes:
            return Vector3.zeros(pos.shape, mask=pos.mask)

        # Generate info about intercept points
        (x,y,z) = pos.to_scalars(recursive=False)
        radius = (x**2 + y**2).sqrt()
        r_vector = Vector3.from_scalars(x,y,0.)

        # Handle radial modes
        if self.modes:
            lon = y.arctan2(x)
            (offset, dr_dt, dlon_dt) = self._mode_offset(lon, time, rates=True)
            a = radius - offset

            if self.gravity:
                dlon_dt += Scalar.minimum(self.gravity.n(a.vals), self.max_rate)

            v_radial = (dr_dt / radius) * r_vector
            v_angular = dlon_dt * Vector3.ZAXIS.cross(r_vector)
            vflat = v_radial + v_angular

        # Handle simple gravity
        else:
            a = radius
            n = Scalar.minimum(self.gravity.n(a.vals), self.max_rate)
            vflat = n * Vector3.ZAXIS.cross(r_vector)

        # The velocity is undefined outside the ring's radial limits
        if self.radii is not None:
            mask = (a < self.radii[0]) | (a > self.radii[1])
            if np.any(mask):
                vflat = vflat.remask_or(mask)

        return vflat

    ############################################################################
    # Radius conversions
    ############################################################################

    def _mode_offset(self, lon, time, derivs=False, rates=False):
        """Sum of the modes as a local radial offset from the mean epicyclic
        radius to the actual radius.

        If input rates==True, return a tuple (radial offset, epicyclic dr/dt,
        epicyclic dlon/dt.
        """

        offset = 0.
        dr_dt = 0.
        dlon_dt = 0.
        for mode in self.modes:
            (cycles, amp, peri0, speed) = mode
            arg = cycles * (lon - peri0) + speed * (time - self.epoch)
            amp_cos_arg = amp * arg.cos(recursive=derivs)
            offset = offset - amp_cos_arg
            if rates:
                dr_dt   = dr_dt   + (speed * amp) * arg.sin()
                dlon_dt = dlon_dt + (2. * speed) * amp_cos_arg

        if rates:
            return (offset, dr_dt, dlon_dt)

        return offset

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
        pos = Vector3(pos)

        # No gravity, no modes
        refplane = RingPlane(Path.SSB, Frame.J2000)

        vels = refplane.velocity(obs)
        self.assertEqual(vels, (0.,0.,0.))

        # No gravity, motionless mode
        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 0.)], epoch=0.)

        vels = plane.velocity(obs)
        self.assertEqual(vels, (0.,0.,0.))

        # No gravity, modes (10 cycles, 100 km amplitude, period = 10,000 s)
        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 100., 0., 2.*np.pi/1.e4)], epoch=0.)

        TIME = 0.
        (a0,theta0) = plane.coords_from_vector3(pos, time=TIME - 0.5)
        (a ,theta ) = plane.coords_from_vector3(pos, time=TIME)
        (a1,theta1) = plane.coords_from_vector3(pos, time=TIME + 0.5)
        self.assertEqual(theta, theta0)
        self.assertEqual(theta, theta1)

        vels = plane.velocity(pos, time=TIME)
        v_angular = vels.perp(pos)
        v_radial = vels - v_angular

        sign = v_radial.dot(pos).sign()
        speed2 = sign * v_radial.norm()
        speed1 = a0 - a1
        self.assertTrue(abs(speed1 - speed2).max() < 2.e-9)

        # Gravity, no modes
        plane = RingPlane(Path.SSB, Frame.J2000, gravity=Gravity.SATURN)

        (a, theta) = plane.coords_from_vector3(pos)

        vels = plane.velocity(pos)
        sep = vels.sep(pos)
        self.assertTrue(abs(sep - np.pi/2.).max() < 1.e-14)

        speed1 = vels.norm()
        rate = np.minimum(Gravity.SATURN.n(a.vals), plane.max_rate)
        diff = (a * rate - speed1) / speed1
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
