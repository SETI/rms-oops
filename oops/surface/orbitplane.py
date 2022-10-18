################################################################################
# oops/surface/orbitplane.py: OrbitPlane subclass of class Surface
################################################################################

import numpy as np
from polymath import Scalar, Vector3

from .                     import Surface
from .ringplane            import RingPlane
from ..event               import Event
from ..frame               import Frame
from ..frame.inclinedframe import InclinedFrame
from ..frame.spinframe     import SpinFrame
from ..path                import Path
from ..path.circlepath     import CirclePath
from ..constants           import PI, HALFPI, TWOPI, RPD

class OrbitPlane(Surface):
    """A subclass of the Surface class describing a flat surface sharing its
    geometric center and tilt with a body on an eccentric and/or inclined orbit.
    The orbit is described as circle offset from the center of the planet by a
    distance ae; this approximation is only accurate to first order in
    eccentricty.

    The coordinate system consists of cylindrical coordinates (a, theta, z)
    where a is the mean radius of the orbit. The zero of longitude is aligned
    with the pericenter.

    The system is masked outside the semimajor axis, but unmasked inside.
    However, coordinates and intercepts are calculated at all locations.
    """

    COORDINATE_TYPE = "polar"
    IS_VIRTUAL = False

    #===========================================================================
    def __init__(self, elements, epoch, origin, frame, path_id=None):
        """Constructor for an OrbitPlane surface.

            elements    a tuple containing three, six or nine orbital elements:
                a           mean radius of orbit, km.
                lon         mean longitude at epoch of a reference object, in
                            radians. This is provided if the user wishes to
                            track a moving body in the plane. However, it does
                            not affect the surface or its coordinate system.
                n           mean motion of a body orbiting within the ring, in
                            radians/sec. This affects velocities returned by
                            the surface but not the surface or its coordinate
                            system.

                e           orbital eccentricty.
                peri        longitude of pericenter at epoch, radians.
                prec        pericenter precession rate, radians/sec.

                i           inclination, radians.
                node        longitude of ascending node at epoch, radians.
                regr        nodal regression rate, radians/sec, NEGATIVE!

            epoch       the time TDB relative to which all orbital elements are
                        defined.
            origin      the path or ID of the planet center.
            frame       the frame or ID of the frame in which the orbit is
                        defined. Should be inertial.
            path_id     the ID under which to register the orbit path; None to
                        leave it unregistered

        Note that the origin and frame used by the returned OrbitPlane object
        will differ from those used to define it here.
        """

        # Save the initial center path and frame. The frame should be inertial.
        self.defined_origin = Path.as_waypoint(origin)
        self.defined_frame  = Frame.as_wayframe(frame)
        assert self.defined_frame.origin is None    # assert inertial

        # We will update the surface's actual path and frame as needed
        self.internal_origin = self.defined_origin
        self.internal_frame  = self.defined_frame

        # Save the orbital elements
        self.a   = elements[0]
        self.lon = elements[1]
        self.n   = elements[2]

        self.epoch = Scalar.as_scalar(epoch)

        # Interpret the inclination
        self.has_inclination = (len(elements) >= 9)
        if self.has_inclination:
            self.i = elements[6]
            self.has_inclination = (self.i != 0)

        # If the orbit is inclined, define a special-purpose inclined frame
        if self.has_inclination:
            if path_id is None:
                frame_id = None
            else:
                frame_id = path_id + "_INCLINATION"

            self.inclined_frame = InclinedFrame(elements[6],  # inclination
                                                elements[7],  # ascending node
                                                elements[8],  # regression rate
                                                self.epoch,
                                                self.internal_frame,
                                                True,         # despin
                                                frame_id = frame_id)
            self.internal_frame = self.inclined_frame
        else:
            self.inclined_frame = None

        # The inclined frame changes its tilt relative to the equatorial plane,
        # accounting for nodal regression, but does not change the reference
        # longitude from that used by the initial frame.

        # Interpret the eccentricity
        self.has_eccentricity = (len(elements) >= 6)
        if self.has_eccentricity:
            self.e = elements[3]
            self.has_eccentricity = (self.e != 0)

        # If the orbit is eccentric, construct a special-purpose path defining
        # the center of the displaced ring
        if self.has_eccentricity:
            self.ae = self.a * self.e
            self.lon_sub_peri = self.lon - elements[4]
            self.n_sub_prec = self.n - elements[5]

            if path_id is None:
                new_path_id = None
            else:
                new_path_id = path_id + "_ECCENTRICITY"

            self.peri_path = CirclePath(elements[0] * elements[3],  # a*e
                                        elements[4] + PI,           # apocenter
                                        elements[5],                # precession
                                        self.epoch,                 # epoch
                                        self.internal_origin,       # origin
                                        self.internal_frame,        # reference
                                        path_id = new_path_id)
            self.internal_origin = self.peri_path

            # The peri_path circulates around the initial origin but does not
            # rotate.

            if path_id is None:
                frame_id = None
            else:
                frame_id = path_id + "_PERICENTER"

            self.spin_frame = SpinFrame(elements[4],                # pericenter
                                        elements[5],                # precession
                                        self.epoch,                 # epoch
                                        2,                          # z-axis
                                        self.internal_frame,        # reference
                                        frame_id = frame_id)
            self.internal_frame = self.spin_frame

        else:
            self.peri_path = None
            self.spin_frame = None

        self.ringplane = RingPlane(self.internal_origin,
                                   self.internal_frame,
                                   radii=None, gravity=None, elevation=0.)

        # The primary origin and frame for the orbit
        self.origin = self.internal_origin.waypoint
        self.frame = self.internal_frame.wayframe

    def __getstate__(self):
        return (self.elements, self.epoch, self.origin, self.frame)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        return self.ringplane.coords_from_vector3(pos, obs, axes=axes,
                                                       derivs=derivs)

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        return self.ringplane.vector3_from_coords(coords, obs, derivs=derivs)

    #===========================================================================
    def intercept(self, obs, los, time=None, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       initial guess at the t array, optional.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            position = obs + t * los
        """

        return self.ringplane.intercept(obs, los, derivs=derivs, guess=guess)

    #===========================================================================
    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        return self.ringplane.normal(pos, derivs=derivs)

    #===========================================================================
    def velocity(self, pos, time=None):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.

        Return:         a Vector3 of velocities, in units of km/s.
        """

        if self.has_eccentricity:
            # For purposes of a first-order velocity calculation, we can assume
            # that the difference between mean longitude and true longitude, in
            # a planet-centered frame, is small.
            #
            # In an inertial, planet-centered frame:
            #
            # r = a - ae cos(lon - peri)
            # lon = lon0 + n * (time - epoch) + 2ae sin(lon - peri)
            #
            # dr/dt = ae sin(lon - peri) (n - prec)
            # dlon/dt = n + 2ae cos(n - peri) (n - prec)
            #
            # In a frame rotating at rate = prec:
            #
            # dr/dt = ae sin(lon - peri) (n - prec)
            # dlon/dt = (n - prec) + 2ae cos(lon - peri) (n - prec)
            #
            # x = r cos(lon)
            # y = r sin(lon)
            #
            # dx/dt = dr/dt * cos(lon) - r sin(lon) dlon/dt
            # dy/dy = dr/dt * sin(lon) + r cos(lon) dlon/dt

            (x,y,z) = pos.to_scalars()
            x = x + self.ae         # shift origin to center of planet

            r = (x**2 + y**2).sqrt()
            cos_lon_sub_peri = x/r
            sin_lon_sub_peri = y/r

            dr_dt = sin_lon_sub_peri * (self.ae * self.n_sub_prec)
            r_dlon_dt = r * self.n_sub_prec * (cos_lon_sub_peri * 2*self.ae + 1)

            dx_dt = dr_dt * cos_lon_sub_peri - r_dlon_dt * sin_lon_sub_peri
            dy_dt = dr_dt * sin_lon_sub_peri + r_dlon_dt * cos_lon_sub_peri

            return Vector3.from_scalars(dx_dt, dy_dt, 0.)

        else:
            return self.n * Vector3.ZAXIS.cross(pos)

    ############################################################################
    # Longitude-anomaly conversions
    ############################################################################

    def from_mean_anomaly(self, anom):
        """The longitude in this frame based on the mean anomaly.

        Accurate to first order in eccentricity.
        """

        anom = Scalar.as_scalar(anom)

        if not self.has_eccentricity:
            return anom
        else:
            return anom + (2*self.ae) * anom.sin()

    #===========================================================================
    def to_mean_anomaly(self, lon):
        """The mean anomaly given an orbital longitude.

        Accurate to first order in eccentricity. Iteration is performed using
        Newton's method to ensure that this function is an exact inverse of
        from_mean_anomaly().
        """

        lon = Scalar.as_scalar(lon)
        if not self.has_eccentricity:
            return lon

        # Solve lon = x + 2ae sin(x)
        #
        # Let
        #   y(x) = x + 2ae sin(x) - lon
        #
        #   dy/dx = 1 + 2ae cos(x)
        #
        # For x[n] as a guess at n,
        #   x[n+1] = x[n] - y(x[n]) / dy/dx

        ae_x2 = 2 * self.ae
        x = lon - ae_x2 * lon.sin()

        # Iterate until all improvement ceases. Should not take long
        prev_max_abs_dx = TWOPI
        max_abs_dx = PI
        while (max_abs_dx < prev_max_abs_dx):
            dx = (lon - x - ae_x2 * x.sin()) / (x.cos() * ae_x2 + 1)
            x += dx

            prev_max_abs_dx = max_abs_dx
            max_abs_dx = abs(dx).max()

        return x

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_OrbitPlane(unittest.TestCase):

    def runTest(self):

        # elements = (a, lon, n)

        # Circular orbit, no derivatives, forward
        elements = (1, 0, 1)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000", "TEST")

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, None, axes=3, derivs=False)

        r_true = Scalar([1,2,1,1])
        l_true = Scalar([0, 0, PI, HALFPI])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true).max() < 1.e-12)
        self.assertTrue(abs(l - l_true).max() < 1.e-12)
        self.assertTrue(abs(z - z_true).max() < 1.e-12)

        # Circular orbit, no derivatives, reverse
        pos2 = orbit.vector3_from_coords((r, l, z), None, derivs=False)

        self.assertTrue((pos - pos2).norm().max() < 1.e-10)

        # Circular orbit, with derivatives, forward
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        pos.insert_deriv('pos', Vector3.IDENTITY, override=True)
        eps = 1.e-6
        delta = 1.e-4

        for step in ([eps,0,0], [0,eps,0], [0,0,eps]):
            dpos = Vector3(step)
            (r,l,z) = orbit.coords_from_vector3(pos + dpos, None, axes=3,
                                                derivs=True)

            r_test = r + r.d_dpos.chain(dpos)
            l_test = l + l.d_dpos.chain(dpos)
            z_test = z + z.d_dpos.chain(dpos)

            self.assertTrue(abs(r - r_test).max() < delta)
            self.assertTrue(abs(l - l_test).max() < delta)
            self.assertTrue(abs(z - z_test).max() < delta)

        # Circular orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, None, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        r.insert_deriv('r', Scalar.ONE, override=True)
        l.insert_deriv('l', Scalar.ONE, override=True)
        z.insert_deriv('z', Scalar.ONE, override=True)
        pos0 = orbit.vector3_from_coords((r, l, z), None, derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), None, derivs=False)
        pos1_test = pos0 + eps * pos0.d_dr
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), None, derivs=False)
        pos1_test = pos0 + eps * pos0.d_dl
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), None, derivs=False)
        pos1_test = pos0 + eps * pos0.d_dz
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        # elements = (a, lon, n, e, peri, prec)

        # Eccentric orbit, no derivatives, forward
        ae = 0.1
        prec = 0.1
        elements = (1, 0, 1, ae, 0, prec)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000", "TEST")
        eps = 1.e-6
        delta = 1.e-5

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        event = Event(0., pos, "SSB", "J2000")
        (r,l,z) = orbit.coords_of_event(event, derivs=False)

        r_true = Scalar([1. + ae, 2. + ae, 1 - ae, np.sqrt(1. + ae**2)])
        l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,ae)])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true).max() < delta)
        self.assertTrue(abs(l - l_true).max() < delta)
        self.assertTrue(abs(z - z_true).max() < delta)

        # Eccentric orbit, no derivatives, reverse
        event2 = orbit.event_at_coords(event.time, (r,l,z)).wrt_ssb()
        self.assertTrue((pos - event2.pos).norm().max() < 1.e-10)
        self.assertTrue((event2.vel).norm().max() < 1.e-10)

        # Eccentric orbit, with derivatives, forward
        ae = 0.1
        prec = 0.1
        elements = (1, 0, 1, ae, 0, prec)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000")
        eps = 1.e-6
        delta = 3.e-5

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])

        for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
            vel = Vector3(v)
            event = Event(0., (pos, vel), "SSB", "J2000")
            (r,l,z) = orbit.coords_of_event(event, derivs=True)

            event = Event(eps, (pos + vel*eps, vel), "SSB", "J2000")
            (r1,l1,z1) = orbit.coords_of_event(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = (l1 - l) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).max() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).max() < delta)

            d_dl_dt = ((l.d_dt*eps - dl_dt_test*eps + PI) % TWOPI - PI) / eps
            self.assertTrue(abs(d_dl_dt).max() < delta)

        # Eccentric orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        r.insert_deriv('r', Scalar.ONE)
        l.insert_deriv('l', Scalar.ONE)
        z.insert_deriv('z', Scalar.ONE)
        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dr
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dl
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dz
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        # elements = (a, lon, n, e, peri, prec, i, node, regr)

        # Inclined orbit, no eccentricity, no derivatives, forward
        inc = 0.1
        regr = -0.1
        node = -HALFPI
        sini = np.sin(inc)
        cosi = np.cos(inc)

        elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000")
        eps = 1.e-6
        delta = 1.e-5

        dz = 0.1
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])
        event = Event(0., pos, "SSB", "J2000")
        (r,l,z) = orbit.coords_of_event(event, derivs=False)

        r_true = Scalar([cosi, 2*cosi, cosi, np.sqrt(1 + (dz*sini)**2)])
        l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,dz*sini)])
        z_true = Scalar([-sini, -2*sini, sini, dz*cosi])

        self.assertTrue(abs(r - r_true).max() < delta)
        self.assertTrue(abs(l - l_true).max() < delta)
        self.assertTrue(abs(z - z_true).max() < delta)

        # Inclined orbit, no derivatives, reverse
        event2 = orbit.event_at_coords(event.time, (r,l,z)).wrt_ssb()
        self.assertTrue((pos - event2.pos).norm().max() < 1.e-10)
        self.assertTrue(event2.vel.norm().max() < 1.e-10)

        # Inclined orbit, with derivatives, forward
        inc = 0.1
        regr = -0.1
        node = -HALFPI
        sini = np.sin(inc)
        cosi = np.cos(inc)

        elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000")
        eps = 1.e-6
        delta = 1.e-5

        dz = 0.1
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])

        for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
            vel = Vector3(v)
            event = Event(0., (pos, vel), "SSB", "J2000")
            (r,l,z) = orbit.coords_of_event(event, derivs=True)

            event = Event(eps, (pos + vel*eps, vel), "SSB", "J2000")
            (r1,l1,z1) = orbit.coords_of_event(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = ((l1 - l + PI) % TWOPI - PI) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).max() < delta)
            self.assertTrue(abs(l.d_dt - dl_dt_test).max() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).max() < delta)

        # Inclined orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        r.insert_deriv('r', Scalar.ONE)
        l.insert_deriv('l', Scalar.ONE)
        z.insert_deriv('z', Scalar.ONE)
        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dr
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dl
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dz
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        # From/to mean anomaly
        elements = (1, 0, 1, 0.1, 0, 0.1)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000", "TEST")

        l = np.arange(361) * RPD
        anoms = orbit.to_mean_anomaly(l)

        lons = orbit.from_mean_anomaly(anoms)
        self.assertTrue(abs(lons - l).max() < 1.e-15)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
