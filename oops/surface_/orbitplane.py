################################################################################
# oops/surface_/orbitplane.py: OrbitPlane subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.event              import Event
from oops.surface_.surface   import Surface
from oops.surface_.ringplane import RingPlane

import oops.frame_    as frame_
import oops.path_     as path_
import oops.registry  as registry

from oops.constants import *

class OrbitPlane(Surface):
    """OrbitPlane is a subclass of the Surface class describing a flat surface
    sharing its geometric center and tilt with a body on an eccentric and/or
    inclined orbit. The orbit is described as circle offset from the center of
    the planet by a distance ae; this approximation is only accurate to first
    order in eccentricty.

    The coordinate system consists of cylindrical coordinates (a, theta, z)
    where a is the mean radius of the orbit. The zero of longitude is aligned
    with the pericenter.

    The system is masked outside the semimajor axis, but unmasked inside.
    However, coordinates and intercepts are calculated at all locations.
    """

    COORDINATE_TYPE = "polar"
    IS_VIRTUAL = False

    def __init__(self, elements, epoch, origin, frame, id=None):
        """Constructor for an OffsetPlane surface.

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
            id          the name under which to register a temporary path or
                        frame if it is needed. Not used for circular, equatorial
                        orbits. None to use temporary path and frame names.

        Note that the origin_id and frame_id used by the returned OrbitPlane
        object will differ from those used to define it here. 
        """

        # Save the initial center path and frame. The frame should be inertial.
        self.defined_origin_id = registry.as_path_id(origin)
        self.defined_frame_id = registry.as_frame_id(frame)

        # We will update the surface's actual path and frame as needed
        self.internal_origin_id = self.defined_origin_id
        self.internal_frame_id = self.defined_frame_id

        # Save the orbital elements
        self.a   = elements[0]
        self.lon = elements[1]
        self.n   = elements[2]

        self.epoch = Scalar.as_scalar(epoch)

        # Interpret the inclination
        self.has_inclination = len(elements) >= 9
        if self.has_inclination:
            self.i = elements[6]
            self.has_inclination = (self.i != 0)

        # If the orbit is inclined, define a special-purpose inclined frame
        if self.has_inclination:
            if id is None:
                self.inclined_frame_id = registry.temporary_frame_id()
            else:
                self.inclined_frame_id = id + "_INCLINATION"

            self.inclined_frame = frame_.InclinedFrame(
                                                elements[6],  # inclination
                                                elements[7],  # ascending node
                                                elements[8],  # regression rate
                                                self.epoch,
                                                self.internal_frame_id,
                                                True,         # despin
                                                self.inclined_frame_id)
            self.internal_frame_id = self.inclined_frame_id
        else:
            self.inclined_frame = None
            self.inclined_frame_id = self.internal_frame_id

        # The inclined frame changes its tilt relative to the equatorial plane,
        # accounting for nodal regression, but does not change the reference
        # longitude from that used by the initial frame.

        # Interpret the eccentricity
        self.has_eccentricity = len(elements) >= 6
        if self.has_eccentricity:
            self.e = elements[3]
            self.has_eccentricity = (self.e != 0)

        # If the orbit is eccentric, construct a special-purpose path defining
        # the center of the displaced ring
        if self.has_eccentricity:
            self.ae = self.a * self.e
            self.lon_sub_peri = self.lon - elements[4]
            self.n_sub_prec = self.n - elements[5]

            if id is None:
                self.peri_path_id = registry.temporary_path_id()
            else:
                self.peri_path_id = id + "_ECCENTRICITY"

            self.peri_path = path_.CirclePath(
                                    elements[0] * elements[3],  # a*e
                                    elements[4] + PI,           # apocenter
                                    elements[5],                # precession
                                    self.epoch,                 # epoch
                                    self.internal_origin_id,    # origin
                                    self.internal_frame_id,     # reference
                                    self.peri_path_id)          # id
            self.internal_origin_id = self.peri_path_id

            # The peri_path circulates around the initial origin but does not
            # rotate.

            if id is None:
                self.spin_frame_id = registry.temporary_frame_id()
            else:
                self.spin_frame_id = id + "_PERICENTER"

            self.spin_frame = frame_.SpinFrame(elements[4],     # pericenter
                                            elements[5],        # precession
                                            self.epoch,         # epoch
                                            2,                  # z-axis
                                            self.internal_frame_id, # reference
                                            self.spin_frame_id)     # id
            self.internal_frame_id = self.spin_frame_id
        else:
            self.peri_path = None
            self.peri_path_id = self.internal_origin_id

            self.spin_frame = None
            self.spin_frame_id = self.internal_frame_id

        self.ringplane = RingPlane(self.internal_origin_id,
                                   self.internal_frame_id,
                                   radii=(0,self.a), gravity=None, elevation=0.)

        self.refplane = RingPlane(self.defined_origin_id,
                                  self.defined_frame_id, 
                                  radii=(0,self.a), gravity=None, elevation=0.)

        # The world needs to see the internal frame and path for intercept() to
        # work properly
        self.origin_id = self.internal_origin_id
        self.frame_id = self.internal_frame_id

    def coords_from_vector3(self, pos, axes=2, obs=None, derivs=False):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            obs         ignored.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on an coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        Where derivs is True, then the coordinate has subfield
                        "d_dpos", which contains the partial derivatives with
                        respect to the surface position, represented as a
                        MatrixN object with item shape [1,3].
        """

        return self.ringplane.coords_from_vector3(pos, axes=axes,
                                                       derivs=derivs)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                r       a Scalar of radius values, with optional units.
                theta   a Scalar of pericenter values, with optional units.
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

        return self.ringplane.vector3_from_coords(coords, obs, derivs=derivs)

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

        return self.ringplane.intercept(obs, los, derivs=derivs)

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

        return self.ringplane.normal(pos, derivs=derivs)

    def velocity(self, pos):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet. In this case, it defines the velocity of
        an orbiting object.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
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
            x = x + self.ae        # shift origin to center of planet

            r = (x**2 + y**2).sqrt()
            cos_lon_sub_peri = x/r
            sin_lon_sub_peri = y/r

            dr_dt = (self.ae * self.n_sub_prec) * sin_lon_sub_peri
            r_dlon_dt = self.n_sub_prec * r * (1 + 2*self.ae * cos_lon_sub_peri)

            dx_dt = dr_dt * cos_lon_sub_peri - r_dlon_dt * sin_lon_sub_peri
            dy_dt = dr_dt * sin_lon_sub_peri + r_dlon_dt * cos_lon_sub_peri

            return Vector3.from_scalars(dx_dt, dy_dt, 0.)

        else:
            return self.n * pos.cross((0,0,-1))

    ############################################################################
    # Longitude-anomaly conversions
    ############################################################################

    def from_mean_anomaly(self, anom):
        """Returns the longitude in this coordinate frame based on the mean
        anomaly, and accurate to first order in eccentricity."""

        anom = Scalar.as_standard(anom)

        if not self.has_eccentricity:
            return anom
        else:
            return anom + (2*self.ae) * anom.sin()

    def to_mean_anomaly(self, lon):
        """Returns the mean anomaly given a longitude in this frame, accurate
        to first order in eccentricity. Iteration is performed using Newton's
        method to ensure that this function is an exact inverse of
        from_mean_anomaly().
        """

        lon = Scalar.as_standard(lon)
        if not self.has_eccentricity: return lon

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
            dx = (lon - x - ae_x2 * x.sin()) / (1 + ae_x2 * x.cos())
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
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)

        r_true = Scalar([1,2,1,1])
        l_true = Scalar([0, 0, PI, HALFPI])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true) < 1.e-12)
        self.assertTrue(abs(l - l_true) < 1.e-12)
        self.assertTrue(abs(z - z_true) < 1.e-12)

        # Circular orbit, no derivatives, reverse
        pos2 = orbit.vector3_from_coords((r, l, z), derivs=False)

        self.assertTrue(abs(pos - pos2) < 1.e-10)

        # Circular orbit, with derivatives, forward
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        eps = 1.e-6
        delta = 1.e-4

        for step in ([eps,0,0], [0,eps,0], [0,0,eps]):
            dpos = Vector3(step)
            (r,l,z) = orbit.coords_from_vector3(pos + dpos, axes=3,
                                                derivs=True)

            r_test = r + (r.d_dpos * dpos.as_column()).as_scalar()
            l_test = l + (l.d_dpos * dpos.as_column()).as_scalar()
            z_test = z + (z.d_dpos * dpos.as_column()).as_scalar()

            self.assertTrue(abs(r - r_test) < delta)
            self.assertTrue(abs(l - l_test) < delta)
            self.assertTrue(abs(z - z_test) < delta)

        # Circular orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(0)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(1)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(2)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

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
        event = Event(0., pos, Vector3((0,0,0)), "SSB", "J2000")
        (r,l,z) = orbit.event_as_coords(event, derivs=False)

        r_true = Scalar([1. + ae, 2. + ae, 1 - ae, np.sqrt(1. + ae**2)])
        l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,ae)])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true) < delta)
        self.assertTrue(abs(l - l_true) < delta)
        self.assertTrue(abs(z - z_true) < delta)

        # Eccentric orbit, no derivatives, reverse
        event2 = orbit.coords_as_event(event.time, (r,l,z))
        self.assertTrue(abs(pos - event.pos) < 1.e-10)
        self.assertTrue(abs(event.vel) < 1.e-10)

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
            event = Event(0., pos, vel, "SSB", "J2000")
            (r,l,z) = orbit.event_as_coords(event, derivs=True)

            event = Event(eps, pos + vel*eps, vel, "SSB", "J2000")
            (r1,l1,z1) = orbit.event_as_coords(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = (l1 - l) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).unmasked() < delta)
            self.assertTrue(abs(l.d_dt - dl_dt_test).unmasked() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).unmasked() < delta)

        # Eccentric orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(0)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(1)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(2)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

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
        event = Event(0., pos, Vector3((0,0,0)), "SSB", "J2000")
        (r,l,z) = orbit.event_as_coords(event, derivs=False)

        r_true = Scalar([cosi, 2*cosi, cosi, np.sqrt(1 + (dz*sini)**2)])
        l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,dz*sini)])
        z_true = Scalar([-sini, -2*sini, sini, dz*cosi])

        self.assertTrue(abs(r - r_true) < delta)
        self.assertTrue(abs(l - l_true) < delta)
        self.assertTrue(abs(z - z_true) < delta)

        # Inclined orbit, no derivatives, reverse
        event2 = orbit.coords_as_event(event.time, (r,l,z))
        self.assertTrue(abs(pos - event.pos) < 1.e-10)
        self.assertTrue(abs(event.vel) < 1.e-10)

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
            event = Event(0., pos, vel, "SSB", "J2000")
            (r,l,z) = orbit.event_as_coords(event, derivs=True)

            event = Event(eps, pos + vel*eps, vel, "SSB", "J2000")
            (r1,l1,z1) = orbit.event_as_coords(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = ((l1 - l + PI) % TWOPI - PI) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).unmasked() < delta)
            self.assertTrue(abs(l.d_dt - dl_dt_test).unmasked() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).unmasked() < delta)

        # Inclined orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(0)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(1)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(2)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        # From/to mean anomaly
        elements = (1, 0, 1, 0.1, 0, 0.1)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000", "TEST")

        l = np.arange(361) * RPD
        anoms = orbit.to_mean_anomaly(l)

        lons = orbit.from_mean_anomaly(anoms)
        self.assertTrue(abs(lons - l) < 1.e-15)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
