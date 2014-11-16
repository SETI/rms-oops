################################################################################
# oops/surface_/unsquashedlimb.py: UnsquashedLimb subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface   import Surface
from oops.constants          import *

class UnsquashedLimb(Surface):
    """The UnsquashedLimb surface is defined as the locus of points where a
    surface normal from a spheroid or ellipsoid is roughly perpendicular to the
    line of sight. This provides a convenient coordinate system for describing
    cloud features on the limb of a body.

    The coordinates of UnsquashedLimb are:
        z       the elevation above the surface, in km.
        clock   an angle on the sky, measured clockwise from the projected
                direction of the north pole.
        d       offset distance of a point beyond the virtual limb plane along
                the line of sight, in km.

    Coordinates are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    truly the locus of points where elevation equals zero. However, note that,
    with this definition, the gradient of the elevation is not exactly normal
    to the surface.
    """

    COORDINATE_TYPE = "limb"
    IS_VIRTUAL = True

    def __init__(self, ground, limits=None):
        """Constructor for a Limb surface.

        Input:
            ground      the Surface object relative to which limb points are to
                        be defined. It should be a Spheroid or Ellipsoid,
                        optically using Centric or Graphic coordinates.
            limits      an optional single value or tuple defining the absolute
                        numerical limit(s) placed on the limb; values outside
                        this range are masked.
        """

        assert ground.COORDINATE_TYPE == "spherical"
        self.ground = ground
        self.origin = ground.origin
        self.frame  = ground.frame

        if limits is None:
            self.limits = None
        else:
            self.limits = (limits[0], limits[1])

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
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

        pos = Vector3.as_vector3(pos, derivs).element_mul(self.ground.unsquash)
        obs = Vector3.as_vector3(obs, derivs).element_mul(self.ground.unsquash)
        mask = pos.norm() >= obs.norm()

        pos = pos.mask_where(mask)
        obs = obs.mask_where(mask)

        # Solve for the point where the line of sight is perpendicular to the
        # sphere's surface
        #
        # los = pos - obs
        # cept = pos + t * los
        # cept dot los = 0
        #
        # (pos dot los) + t * (los dot los) = 0
        #
        # t = -(pos dot los) / (los dot los)

        los = pos - obs
        neg_t = pos.dot(los) / los.dot(los)

        cept = pos - neg_t * los

        # Place z=0 at the surface
        z = cept.norm() - self.ground.req

        # Define frame with Z toward the origin, north pole in the X-Z plane
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)

        x = cept.dot(x_axis)
        y = cept.dot(y_axis)

        clock = y.arctan2(x) % TWOPI

        if axes == 2:
            return (z, clock)
        else:
            d = neg_t * los.norm()
            return (z, clock, d)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining coordinates.
            obs         position of the observer in the surface frame.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.
        """

        obs = Vector3.as_vector3(obs, derivs).element_mul(self.ground.unsquash)

        z = Scalar.as_scalar(coords[0], derivs)
        clock = Scalar.as_scalar(coords[1], derivs)

        r = z + self.ground.req
        sin_offset = r / obs.norm()
        r_cos_offset = r * (1 - sin_offset**2).sqrt()

        x =  r_cos_offset * clock.cos()
        y =  r_cos_offset * clock.sin()
        z = -r * sin_offset

        zaxis = -obs.unit()
        yaxis = Vector3.ZAXIS.ucross(obs)
        xaxis = yaxis.cross(zaxis)

        cept = x * xaxis + y * yaxis + z * zaxis

        if len(coords) > 2:
            d = Scalar.as_scalar(coords[2], derivs)
            los = cept - obs
            cept += (d / los.norm()) * los

        return cept.element_mul(self.ground.squash)

    def intercept(self, obs, los, derivs=False):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        obs = Vector3.as_vector3(obs, derivs).element_mul(self.ground.unsquash)
        los = Vector3.as_vector3(los, derivs).element_mul(self.ground.unsquash)

        # Solve for the intercept distance where the line of sight is normal to
        # the surface.
        #
        # cept = obs + t * los
        # cept dot los = 0
        #
        # t = (obs dot los) / (los dot los)

        t = -obs.dot(los) / los.norm_sq()
        t = t.mask_where_le(0)
        cept = obs + t * los

        return (cept.element_mul(self.ground.squash), t)

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.
            guess       an initial guess at the coefficient p such that
                            intercept + p * normal = pos
                        for the associated ground surface;
            groundtrack optional value of the groundtrack, as returned by
                        intercept().

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.

        NOTE: Counterintuitively, we define this as the normal for the
        associated ground surface, so that incidence and emission angles work
        out as the user would expect.
        """

        return self.ground.normal(pos, derivs=derivs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Limb(unittest.TestCase):

    def runTest(self):

        from oops.frame_.frame import Frame
        from oops.path_.path import Path

        from oops.surface_.spheroid  import Spheroid
        from oops.surface_.ellipsoid import Ellipsoid

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.

        NPTS = 100

        ####################
        # Exact sphere
        ####################

        ground = Spheroid("SSB", "J2000", (REQ, REQ))
        limb = UnsquashedLimb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t) = limb.intercept(obs, los)
        self.assertTrue(abs(cept.sep(los) - HALFPI).max() < 1.e-10)

        coords = limb.coords_from_vector3(cept, obs, axes=3)
        self.assertTrue(abs(coords[2]).max() < 1.e-8)
        self.assertTrue((coords[0] > -REQ).all())

        pos = limb.vector3_from_coords(coords[:2], obs)
        self.assertTrue((pos - cept).norm().max() < 1.e-8)

        z = np.random.random(NPTS) * REQ + REQ/2.
        clock = np.random.random(NPTS) * TWOPI
        d = 0.

        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 2*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        pos = limb.vector3_from_coords((z,clock,d), obs)
        coords = limb.coords_from_vector3(pos, obs, axes=3)

        self.assertTrue(abs(coords[0] - z).max() < 1.e-6)
        self.assertTrue(abs(coords[1] - clock).max() < 1.e-10)
        self.assertTrue(abs(coords[2] - d).max() < 1.e-6)

        d = np.random.random(NPTS) * REQ - REQ/2.

        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 2*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        pos = limb.vector3_from_coords((z,clock,d), obs)
        coords = limb.coords_from_vector3(pos, obs, axes=3)

        self.assertTrue(abs(coords[0] - z).max() < 1.e-6)
        self.assertTrue(abs(coords[1] - clock).max() < 1.e-10)
        self.assertTrue(abs(coords[2] - d).max() < 1.e-4)

        # Derivatives of limb.intercept()

        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 2*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        z = Scalar(np.random.random(NPTS) * REQ + REQ/2.)
        clock = Scalar(np.random.random(NPTS) * TWOPI)
        d = Scalar(np.random.random(NPTS) * REQ - REQ/2.)
        pos = limb.vector3_from_coords((z,clock,d), obs)

        los = pos - obs

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)

        (cept,t) = limb.intercept(obs2, los, derivs=True)
        dcept1_dobs = cept.d_dobs.join_items(Matrix).column_vectors()
        dt1_dobs = t.d_dobs.join_items(Vector).to_scalars()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = limb.intercept(obs, los, derivs=False)
            (cept1,t1) = limb.intercept(obs + dobs[i], los, derivs=False)
            (cept2,t2) = limb.intercept(obs - dobs[i], los, derivs=False)

            dcept_dobs = (cept1 - cept2) / (2*eps)
            mean = abs(dcept_dobs).mean()
            self.assertTrue(abs(dcept1_dobs[i] - dcept_dobs).max() < 1.e-3 * mean)

            dt_dobs = (t1 - t2) / (2*eps)
            mean = abs(dt_dobs).mean()
            self.assertTrue(abs(dt1_dobs[i] - dt_dobs).max() < 1.e-3 * mean)

        los2 = los.copy()
        los2.insert_deriv('los', Vector3.IDENTITY)

        (cept,t) = limb.intercept(obs, los2, derivs=True)
        dcept1_dlos = cept.d_dlos.join_items(Matrix).column_vectors()
        dt1_dlos = t.d_dlos.join_items(Vector).to_scalars()

        eps = 1.
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = limb.intercept(obs, los, derivs=False)
            (cept1,t1) = limb.intercept(obs, los + dlos[i], derivs=False)
            (cept2,t2) = limb.intercept(obs, los - dlos[i], derivs=False)

            dcept_dlos = (cept1 - cept2) / (2*eps)
            mean = abs(dcept_dlos).mean()
            self.assertTrue(abs(dcept1_dlos[i] - dcept_dlos).max() < 1.e-3 * mean)

            dt_dlos = (t1 - t2) / (2*eps)
            mean = abs(dt_dlos).mean()
            self.assertTrue(abs(dt1_dlos[i] - dt_dlos).max() < 1.e-3 * mean)

        # Derivatives of limb.vector3_from_coords()

        z2 = z.copy()
        z2.insert_deriv('z', Scalar.ONE)
        pos = limb.vector3_from_coords((z2, clock, d), obs, derivs=True)
        dpos1_dz = pos.d_dz

        dz = 1.
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z+dz, clock, d), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z-dz, clock, d), obs, derivs=False)

        dpos_dz = (pos1 - pos2) / (2*dz)
        mean = abs(dpos1_dz).mean()
        self.assertTrue(abs(dpos1_dz - dpos_dz).max() < 1.e-3 * mean)

        clock2 = clock.copy()
        clock2.insert_deriv('clock', Scalar.ONE)
        pos = limb.vector3_from_coords((z, clock2, d), obs, derivs=True)
        dpos1_dclock = pos.d_dclock

        dc = 1.e-5
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z, clock+dc, d), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z, clock-dc, d), obs, derivs=False)

        dpos_dclock = (pos1 - pos2) / (2*dc)
        self.assertTrue(abs(dpos1_dclock - dpos_dclock).max() < 1.e-5)

        d2 = d.copy()
        d2.insert_deriv('d', Scalar.ONE)
        pos = limb.vector3_from_coords((z, clock, d2), obs, derivs=True)
        dpos1_dd = pos.d_dd

        dd = 1.
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z, clock, d+dd), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z, clock, d-dd), obs, derivs=False)

        dpos_dd = (pos1 - pos2) / (2*dd)
        mean = abs(dpos1_dd).mean()
        self.assertTrue(abs(dpos1_dd - dpos_dd).max() < 1.e-3 * mean)

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)
        pos = limb.vector3_from_coords((z, clock, d), obs2, derivs=True)
        dpos1_dobs = pos.d_dobs.join_items(Matrix).column_vectors()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            pos1 = limb.vector3_from_coords((z,clock,d), obs + dobs[i], derivs=False)
            pos2 = limb.vector3_from_coords((z,clock,d), obs - dobs[i], derivs=False)

            dpos_dobs = (pos1 - pos2) / (2*eps)
            mean = abs(dpos_dobs).mean()
            self.assertTrue(abs(dpos1_dobs[i] - dpos_dobs).max() < 1.e-3 * mean)

        # Derivatives of limb.coords_from_vector3()

        z = Scalar(np.random.random(NPTS) * REQ + REQ/2.)
        clock = Scalar(np.random.random(NPTS) * TWOPI)
        d = Scalar(np.random.random(NPTS) * REQ - REQ/2.)
        pos = limb.vector3_from_coords((z,clock,d), obs)

        pos2 = pos.copy()
        pos2.insert_deriv('pos', Vector3.IDENTITY)
        (z, clock, d) = limb.coords_from_vector3(pos2, obs, axes=3, derivs=True)
        dz1_dpos = z.d_dpos.join_items(Vector).to_scalars()
        dclock1_dpos = clock.d_dpos.join_items(Vector).to_scalars()
        dd1_dpos = d.d_dpos.join_items(Vector).to_scalars()

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (z1, clock1, d1) = limb.coords_from_vector3(pos + dpos[i], obs,
                                                        axes=3, derivs=False)
            (z2, clock2, d2) = limb.coords_from_vector3(pos - dpos[i], obs,
                                                        axes=3, derivs=False)

            dz_dpos = (z1 - z2) / (2*eps)
            mean = abs(dz_dpos).mean()
            self.assertTrue(abs(dz1_dpos[i] - dz_dpos).max() < 1.e-3 * mean)


            dclock_dpos = (clock1 - clock2) / (2*eps)
            mean = abs(dclock_dpos).mean()
            self.assertTrue(abs(dclock1_dpos[i] - dclock_dpos).max() < 1.e-3 * mean)

            dd_dpos = (d1 - d2) / (2*eps)
            mean = abs(dd_dpos).mean()
            self.assertTrue(abs(dd1_dpos[i] - dd_dpos).max() < 1.e-3 * mean)

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)
        (z, clock, d) = limb.coords_from_vector3(pos, obs2, axes=3, derivs=True)
        dz1_dobs = z.d_dobs.join_items(Vector).to_scalars()
        dclock1_dobs = clock.d_dobs.join_items(Vector).to_scalars()
        dd1_dobs = d.d_dobs.join_items(Vector).to_scalars()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (z1, clock1, d1) = limb.coords_from_vector3(pos, obs + dobs[i],
                                                        axes=3, derivs=False)
            (z2, clock2, d2) = limb.coords_from_vector3(pos, obs - dobs[i],
                                                        axes=3, derivs=False)

            dz_dobs = (z1 - z2) / (2*eps)
            mean = abs(dz_dobs).mean()
            self.assertTrue(abs(dz1_dobs[i] - dz_dobs).max() < 1.e-3 * mean)


            dclock_dobs = (clock1 - clock2) / (2*eps)
            mean = abs(dclock_dobs).mean()
            self.assertTrue(abs(dclock1_dobs[i] - dclock_dobs).max() < 1.e-3 * mean)


            dd_dobs = (d1 - d2) / (2*eps)
            mean = abs(dd_dobs).mean()
            self.assertTrue(abs(dd1_dobs[i] - dd_dobs).max() < 1.e-3 * mean)

        ####################
        # Spheroid
        ####################

        ground = Spheroid("SSB", "J2000", (REQ, RPOL))
        limb = UnsquashedLimb(ground)

        obs = Vector3([4*REQ,0,0])

        z = np.random.random(NPTS) * REQ + REQ/2.
        clock = np.random.random(NPTS) * TWOPI
        d = np.random.random(NPTS) * REQ - REQ/2.

        pos = limb.vector3_from_coords((z,clock,d), obs)
        coords = limb.coords_from_vector3(pos, obs, axes=3)

        self.assertTrue(abs(coords[0] - z).max() < 1.e-6)
        self.assertTrue(abs(coords[1] - clock).max() < 1.e-10)
        self.assertTrue(abs(coords[2] - d).max() < 1.e-6)

        pos = limb.vector3_from_coords((z,clock), obs)
        los = pos - obs
        (cept, t) = limb.intercept(obs, los)

        self.assertTrue((cept - pos).norm().max() < 1.e-6)

        # Limb points
        pos = limb.vector3_from_coords((0.,clock), obs)
        los = pos - obs

        normal = limb.normal(pos)
        self.assertTrue(abs(los.sep(normal) - HALFPI).max() < 1.e-10)

        # Derivatives of limb.intercept()

        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 2*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        z = Scalar(np.random.random(NPTS) * REQ + REQ/2.)
        clock = Scalar(np.random.random(NPTS) * TWOPI)
        d = Scalar(np.random.random(NPTS) * REQ - REQ/2.)
        pos = limb.vector3_from_coords((z,clock,d), obs)

        los = pos - obs

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)

        (cept,t) = limb.intercept(obs2, los, derivs=True)
        dcept1_dobs = cept.d_dobs.join_items(Matrix).column_vectors()
        dt1_dobs = t.d_dobs.join_items(Vector).to_scalars()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = limb.intercept(obs, los, derivs=False)
            (cept1,t1) = limb.intercept(obs + dobs[i], los, derivs=False)
            (cept2,t2) = limb.intercept(obs - dobs[i], los, derivs=False)

            dcept_dobs = (cept1 - cept2) / (2*eps)
            mean = abs(dcept_dobs).mean()
            self.assertTrue(abs(dcept1_dobs[i] - dcept_dobs).max() < 1.e-3 * mean)

            dt_dobs = (t1 - t2) / (2*eps)
            mean = abs(dt_dobs).mean()
            self.assertTrue(abs(dt1_dobs[i] - dt_dobs).max() < 1.e-3 * mean)

        los2 = los.copy()
        los2.insert_deriv('los', Vector3.IDENTITY)

        (cept,t) = limb.intercept(obs, los2, derivs=True)
        dcept1_dlos = cept.d_dlos.join_items(Matrix).column_vectors()
        dt1_dlos = t.d_dlos.join_items(Vector).to_scalars()

        eps = 1.
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = limb.intercept(obs, los, derivs=False)
            (cept1,t1) = limb.intercept(obs, los + dlos[i], derivs=False)
            (cept2,t2) = limb.intercept(obs, los - dlos[i], derivs=False)

            dcept_dlos = (cept1 - cept2) / (2*eps)
            mean = abs(dcept_dlos).mean()
            self.assertTrue(abs(dcept1_dlos[i] - dcept_dlos).max() < 1.e-3 * mean)

            dt_dlos = (t1 - t2) / (2*eps)
            mean = abs(dt_dlos).mean()
            self.assertTrue(abs(dt1_dlos[i] - dt_dlos).max() < 1.e-3 * mean)

        # Derivatives of limb.vector3_from_coords()

        z2 = z.copy()
        z2.insert_deriv('z', Scalar.ONE)
        pos = limb.vector3_from_coords((z2, clock, d), obs, derivs=True)
        dpos1_dz = pos.d_dz

        dz = 1.
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z+dz, clock, d), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z-dz, clock, d), obs, derivs=False)

        dpos_dz = (pos1 - pos2) / (2*dz)
        mean = abs(dpos1_dz).mean()
        self.assertTrue(abs(dpos1_dz - dpos_dz).max() < 1.e-3 * mean)

        clock2 = clock.copy()
        clock2.insert_deriv('clock', Scalar.ONE)
        pos = limb.vector3_from_coords((z, clock2, d), obs, derivs=True)
        dpos1_dclock = pos.d_dclock

        dc = 1.e-5
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z, clock+dc, d), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z, clock-dc, d), obs, derivs=False)

        dpos_dclock = (pos1 - pos2) / (2*dc)
        self.assertTrue(abs(dpos1_dclock - dpos_dclock).max() < 1.e-5)

        d2 = d.copy()
        d2.insert_deriv('d', Scalar.ONE)
        pos = limb.vector3_from_coords((z, clock, d2), obs, derivs=True)
        dpos1_dd = pos.d_dd

        dd = 1.
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z, clock, d+dd), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z, clock, d-dd), obs, derivs=False)

        dpos_dd = (pos1 - pos2) / (2*dd)
        mean = abs(dpos1_dd).mean()
        self.assertTrue(abs(dpos1_dd - dpos_dd).max() < 1.e-3 * mean)

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)
        pos = limb.vector3_from_coords((z, clock, d), obs2, derivs=True)
        dpos1_dobs = pos.d_dobs.join_items(Matrix).column_vectors()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            pos1 = limb.vector3_from_coords((z,clock,d), obs + dobs[i], derivs=False)
            pos2 = limb.vector3_from_coords((z,clock,d), obs - dobs[i], derivs=False)

            dpos_dobs = (pos1 - pos2) / (2*eps)
            mean = abs(dpos_dobs).mean()
            self.assertTrue(abs(dpos1_dobs[i] - dpos_dobs).max() < 1.e-3 * mean)

        # Derivatives of limb.coords_from_vector3()

        z = Scalar(np.random.random(NPTS) * REQ + REQ/2.)
        clock = Scalar(np.random.random(NPTS) * TWOPI)
        d = Scalar(np.random.random(NPTS) * REQ - REQ/2.)
        pos = limb.vector3_from_coords((z,clock,d), obs)

        pos2 = pos.copy()
        pos2.insert_deriv('pos', Vector3.IDENTITY)
        (z, clock, d) = limb.coords_from_vector3(pos2, obs, axes=3, derivs=True)
        dz1_dpos = z.d_dpos.join_items(Vector).to_scalars()
        dclock1_dpos = clock.d_dpos.join_items(Vector).to_scalars()
        dd1_dpos = d.d_dpos.join_items(Vector).to_scalars()

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (z1, clock1, d1) = limb.coords_from_vector3(pos + dpos[i], obs,
                                                        axes=3, derivs=False)
            (z2, clock2, d2) = limb.coords_from_vector3(pos - dpos[i], obs,
                                                        axes=3, derivs=False)

            dz_dpos = (z1 - z2) / (2*eps)
            mean = abs(dz_dpos).mean()
            self.assertTrue(abs(dz1_dpos[i] - dz_dpos).max() < 1.e-3 * mean)

            dclock_dpos = (clock1 - clock2) / (2*eps)
            mean = abs(dclock_dpos).mean()
            self.assertTrue(abs(dclock1_dpos[i] - dclock_dpos).max() < 1.e-3 * mean)

            dd_dpos = (d1 - d2) / (2*eps)
            mean = abs(dd_dpos).mean()
            self.assertTrue(abs(dd1_dpos[i] - dd_dpos).max() < 1.e-3 * mean)

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)
        (z, clock, d) = limb.coords_from_vector3(pos, obs2, axes=3, derivs=True)
        dz1_dobs = z.d_dobs.join_items(Vector).to_scalars()
        dclock1_dobs = clock.d_dobs.join_items(Vector).to_scalars()
        dd1_dobs = d.d_dobs.join_items(Vector).to_scalars()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (z1, clock1, d1) = limb.coords_from_vector3(pos, obs + dobs[i],
                                                        axes=3, derivs=False)
            (z2, clock2, d2) = limb.coords_from_vector3(pos, obs - dobs[i],
                                                        axes=3, derivs=False)

            dz_dobs = (z1 - z2) / (2*eps)
            mean = abs(dz_dobs).mean()
            self.assertTrue(abs(dz1_dobs[i] - dz_dobs).max() < 1.e-3 * mean)


            dclock_dobs = (clock1 - clock2) / (2*eps)
            mean = abs(dclock_dobs).mean()
            self.assertTrue(abs(dclock1_dobs[i] - dclock_dobs).max() < 1.e-3 * mean)


            dd_dobs = (d1 - d2) / (2*eps)
            mean = abs(dd_dobs).mean()
            self.assertTrue(abs(dd1_dobs[i] - dd_dobs).max() < 1.e-3 * mean)

        ####################
        # Ellipsoid
        ####################

        ground = Ellipsoid("SSB", "J2000", (REQ, RMID, RPOL))
        limb = UnsquashedLimb(ground)

        obs = Vector3([4*REQ,0,0])

        z = np.random.random(NPTS) * REQ + REQ/2.
        clock = np.random.random(NPTS) * TWOPI
        d = np.random.random(NPTS) * REQ - REQ/2.

        pos = limb.vector3_from_coords((z,clock,d), obs)
        coords = limb.coords_from_vector3(pos, obs, axes=3)

        self.assertTrue(abs(coords[0] - z).max() < 1.e-6)
        self.assertTrue(abs(coords[1] - clock).max() < 1.e-10)
        self.assertTrue(abs(coords[2] - d).max() < 1.e-6)

        pos = limb.vector3_from_coords((z,clock), obs)
        los = pos - obs
        (cept, t) = limb.intercept(obs, los)

        self.assertTrue((cept - pos).norm().max() < 1.e-6)

        # Limb points
        pos = limb.vector3_from_coords((0.,clock), obs)
        los = pos - obs

        normal = limb.normal(pos)
        self.assertTrue(abs(los.sep(normal) - HALFPI).max() < 1.e-10)

        # Derivatives of limb.intercept()

        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 2*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        z = Scalar(np.random.random(NPTS) * REQ + REQ/2.)
        clock = Scalar(np.random.random(NPTS) * TWOPI)
        d = Scalar(np.random.random(NPTS) * REQ - REQ/2.)
        pos = limb.vector3_from_coords((z,clock,d), obs)

        los = pos - obs

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)

        (cept,t) = limb.intercept(obs2, los, derivs=True)
        dcept1_dobs = cept.d_dobs.join_items(Matrix).column_vectors()
        dt1_dobs = t.d_dobs.join_items(Vector).to_scalars()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = limb.intercept(obs, los, derivs=False)
            (cept1,t1) = limb.intercept(obs + dobs[i], los, derivs=False)
            (cept2,t2) = limb.intercept(obs - dobs[i], los, derivs=False)

            dcept_dobs = (cept1 - cept2) / (2*eps)
            mean = abs(dcept_dobs).mean()
            self.assertTrue(abs(dcept1_dobs[i] - dcept_dobs).max() < 1.e-3 * mean)

            dt_dobs = (t1 - t2) / (2*eps)
            mean = abs(dt_dobs).mean()
            self.assertTrue(abs(dt1_dobs[i] - dt_dobs).max() < 1.e-3 * mean)

        los2 = los.copy()
        los2.insert_deriv('los', Vector3.IDENTITY)

        (cept,t) = limb.intercept(obs, los2, derivs=True)
        dcept1_dlos = cept.d_dlos.join_items(Matrix).column_vectors()
        dt1_dlos = t.d_dlos.join_items(Vector).to_scalars()

        eps = 1.
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = limb.intercept(obs, los, derivs=False)
            (cept1,t1) = limb.intercept(obs, los + dlos[i], derivs=False)
            (cept2,t2) = limb.intercept(obs, los - dlos[i], derivs=False)

            dcept_dlos = (cept1 - cept2) / (2*eps)
            mean = abs(dcept_dlos).mean()
            self.assertTrue(abs(dcept1_dlos[i] - dcept_dlos).max() < 1.e-3 * mean)

            dt_dlos = (t1 - t2) / (2*eps)
            mean = abs(dt_dlos).mean()
            self.assertTrue(abs(dt1_dlos[i] - dt_dlos).max() < 1.e-3 * mean)

        # Derivatives of limb.vector3_from_coords()

        z2 = z.copy()
        z2.insert_deriv('z', Scalar.ONE)
        pos = limb.vector3_from_coords((z2, clock, d), obs, derivs=True)
        dpos1_dz = pos.d_dz

        dz = 1.
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z+dz, clock, d), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z-dz, clock, d), obs, derivs=False)

        dpos_dz = (pos1 - pos2) / (2*dz)
        mean = abs(dpos1_dz).mean()
        self.assertTrue(abs(dpos1_dz - dpos_dz).max() < 1.e-3 * mean)

        clock2 = clock.copy()
        clock2.insert_deriv('clock', Scalar.ONE)
        pos = limb.vector3_from_coords((z, clock2, d), obs, derivs=True)
        dpos1_dclock = pos.d_dclock

        dc = 1.e-5
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z, clock+dc, d), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z, clock-dc, d), obs, derivs=False)

        dpos_dclock = (pos1 - pos2) / (2*dc)
        self.assertTrue(abs(dpos1_dclock - dpos_dclock).max() < 1.e-5)

        d2 = d.copy()
        d2.insert_deriv('d', Scalar.ONE)
        pos = limb.vector3_from_coords((z, clock, d2), obs, derivs=True)
        dpos1_dd = pos.d_dd

        dd = 1.
        pos = limb.vector3_from_coords((z, clock, d), obs, derivs=False)
        pos1 = limb.vector3_from_coords((z, clock, d+dd), obs, derivs=False)
        pos2 = limb.vector3_from_coords((z, clock, d-dd), obs, derivs=False)

        dpos_dd = (pos1 - pos2) / (2*dd)
        mean = abs(dpos1_dd).mean()
        self.assertTrue(abs(dpos1_dd - dpos_dd).max() < 1.e-3 * mean)

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)
        pos = limb.vector3_from_coords((z, clock, d), obs2, derivs=True)
        dpos1_dobs = pos.d_dobs.join_items(Matrix).column_vectors()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            pos1 = limb.vector3_from_coords((z,clock,d), obs + dobs[i], derivs=False)
            pos2 = limb.vector3_from_coords((z,clock,d), obs - dobs[i], derivs=False)

            dpos_dobs = (pos1 - pos2) / (2*eps)
            mean = abs(dpos_dobs).mean()
            self.assertTrue(abs(dpos1_dobs[i] - dpos_dobs).max() < 1.e-3 * mean)

        # Derivatives of limb.coords_from_vector3()

        z = Scalar(np.random.random(NPTS) * REQ + REQ/2.)
        clock = Scalar(np.random.random(NPTS) * TWOPI)
        d = Scalar(np.random.random(NPTS) * REQ - REQ/2.)
        pos = limb.vector3_from_coords((z,clock,d), obs)

        pos2 = pos.copy()
        pos2.insert_deriv('pos', Vector3.IDENTITY)
        (z, clock, d) = limb.coords_from_vector3(pos2, obs, axes=3, derivs=True)
        dz1_dpos = z.d_dpos.join_items(Vector).to_scalars()
        dclock1_dpos = clock.d_dpos.join_items(Vector).to_scalars()
        dd1_dpos = d.d_dpos.join_items(Vector).to_scalars()

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (z1, clock1, d1) = limb.coords_from_vector3(pos + dpos[i], obs,
                                                        axes=3, derivs=False)
            (z2, clock2, d2) = limb.coords_from_vector3(pos - dpos[i], obs,
                                                        axes=3, derivs=False)

            dz_dpos = (z1 - z2) / (2*eps)
            mean = abs(dz_dpos).mean()
            self.assertTrue(abs(dz1_dpos[i] - dz_dpos).max() < 1.e-3 * mean)


            dclock_dpos = (clock1 - clock2) / (2*eps)
            mean = abs(dclock_dpos).mean()
            self.assertTrue(abs(dclock1_dpos[i] - dclock_dpos).max() < 1.e-3 * mean)

            dd_dpos = (d1 - d2) / (2*eps)
            mean = abs(dd_dpos).mean()
            self.assertTrue(abs(dd1_dpos[i] - dd_dpos).max() < 1.e-3 * mean)

        obs2 = obs.copy()
        obs2.insert_deriv('obs', Vector3.IDENTITY)
        (z, clock, d) = limb.coords_from_vector3(pos, obs2, axes=3, derivs=True)
        dz1_dobs = z.d_dobs.join_items(Vector).to_scalars()
        dclock1_dobs = clock.d_dobs.join_items(Vector).to_scalars()
        dd1_dobs = d.d_dobs.join_items(Vector).to_scalars()

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (z1, clock1, d1) = limb.coords_from_vector3(pos, obs + dobs[i],
                                                        axes=3, derivs=False)
            (z2, clock2, d2) = limb.coords_from_vector3(pos, obs - dobs[i],
                                                        axes=3, derivs=False)

            dz_dobs = (z1 - z2) / (2*eps)
            mean = abs(dz_dobs).mean()
            self.assertTrue(abs(dz1_dobs[i] - dz_dobs).max() < 1.e-3 * mean)

            dclock_dobs = (clock1 - clock2) / (2*eps)
            mean = abs(dclock_dobs).mean()
            self.assertTrue(abs(dclock1_dobs[i] - dclock_dobs).max() < 1.e-3 * mean)


            dd_dobs = (d1 - d2) / (2*eps)
            mean = abs(dd_dobs).mean()
            self.assertTrue(abs(dd1_dobs[i] - dd_dobs).max() < 1.e-3 * mean)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
