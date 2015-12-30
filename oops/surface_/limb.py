################################################################################
# oops/surface_/limb.py: Limb subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface   import Surface

from oops.config             import SURFACE_PHOTONS, LOGGING
from oops.constants          import *

class Limb(Surface):
    """The Limb surface is defined as the locus of points where a surface normal
    from a spheroid or ellipsoid is perpendicular to the line of sight. This
    provides a convenient coordinate system for describing cloud features on the
    limb of a body.

    The coordinates of Limb are (lon, lat, z), much the same as for the
    surface of the associated spheroid or ellipsoid. The key difference is in
    how the intercept point is derived.
        lon     longitude at the ground point beneath the limb point, using the
                same definition as that of the associated spheroid or ellipsoid.
        lat     latitude at the ground point beneath the limb point, using the
                same definition as that of the associated spheroid or ellipsoid.
        z       the elevation above the surface, as an actual distance measured
                normal to the ring plane. Note that this definition differs from
                that used by the spheroid and ellipsoid surface.
    """

    COORDINATE_TYPE = "limb"
    IS_VIRTUAL = True
    DEBUG = False   # True for convergence testing in intercept()

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

    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, guess=None, groundtrack=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaulate the surface; ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            guess       an initial guess at the coefficient p such that
                            intercept + p * normal = pos
                        for the associated ground surface;
                        if guess is not None, it appends the converged value of
                        p to the tuple returned;
                        use guess=False; to return the value of p without
                        providing one initially.
            groundtrack True to append a Vector3 of the ground points beneath
                        the given values of ps to the tuple returned.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.

                        If guess is not None, the converged value of p is
                        appended to the returned tuple.

                        if groundtrack is True, a Vector3 of ground points is
                        appended to the returned tuple.
        """

        pos = Vector3.as_vector3(pos, derivs)

        if guess is None:
            ground_guess = False
        else:
            ground_guess = guess

        (track,
         ground_guess) = self.ground.intercept_normal_to(pos, derivs,
                                                         guess=ground_guess)

        (lon, lat) = self.ground.coords_from_vector3(track, derivs)

        if axes == 2:
            results = (lon, lat)
        else:
            z = (pos.norm() - track.norm()).sign() * (pos - track).norm()

            # Mask based on elevation limits if necessary
            if self.limits is not None:
                zmask = (z.vals < self.limits[0]) | (z.vals > self.limits[1])
                lon = lon.mask_where(zmask)
                lat = lat.mask_where(zmask)
                z = z.mask_where(zmask, replace=0.)

            results = (lon, lat, z)

        if guess is not None:
            results += (ground_guess,)

        if groundtrack:
            results += (track,)

        return results

    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False,
                                  groundtrack=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                lon     longitude in radians.
                lat     latitude in radians.
                z       the perpendicular distance from the surface, in km.
            obs         position of the observer in the surface frame.
            time        a Scalar time at which to evaulate the surface; ignored.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.
            groundtrack True to replace the returned value by a tuple, where the
                        second quantity is the groundtrack point as a Vector3.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.
        """

        track = self.ground.vector3_from_coords(coords[:2], derivs=derivs)

        if len(coords) == 2:
            results = (track, track)

        else:
            perp = self.ground.normal(track, derivs=derivs)
            z = Scalar.as_scalar(coords[2], derivs)
            z = z.mask_where(z.mask, replace=0.)

            result = track + (z / perp.norm()) * perp
            results = (result, track)

        if groundtrack:
            return results
        else:
            return results[0]

    def intercept(self, obs, los, time=None, derivs=False, guess=None,
                        groundtrack=False):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            time        a Scalar time at which to evaulate the surface; ignored.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los
            groundtrack True to include the surface intercept points of the body
                        associated with each limb intercept. This array can
                        speed up any subsequent calculations such as calls to
                        normal(), and can be used to determine locations in
                        body coordinates.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        # Convert to standard units
        obs = Vector3.as_vector3(obs, derivs)
        los = Vector3.as_vector3(los, derivs)

        # Solve for the intercept distance where the line of sight is normal to
        # the surface.
        #
        # pos = obs + t * los
        # track = ground.intercept_normal_to(pos(t))
        # normal(track) dot los = 0
        #
        # Solve for t.
        #
        #   f(t) = normal(track(pos(t))) dot los
        #
        #   df/dt = (dnormal/dpos <chain> los) dot los
        #
        # Initial guess is where los and pos are perpendicular:
        # (obs + t * los) dot los = 0
        #
        # t = -(obs dot los) / (los dot los)

        if guess not in (None, False):
            t = guess.copy()
        else:
            t = -obs.dot(los) / los.dot(los)

        max_abs_dt = 1.e99
        ground_guess = False
        for iter in range(SURFACE_PHOTONS.max_iterations):
            pos = obs + t * los
            pos.insert_deriv('pos', Vector3.IDENTITY)

            (track,
             ground_guess) = self.ground.intercept_normal_to(pos, derivs=True,
                                                             guess=ground_guess)
            normal = self.ground.normal(track, derivs=True)

            f = normal.without_derivs().dot(los)
            df_dt = normal.d_dpos.chain(los).dot(los)
            dt = f / df_dt

            t = t - dt

            prev_max_abs_dt = max_abs_dt
            max_abs_dt = abs(dt).max()

            if LOGGING.surface_iterations or Limb.DEBUG:
                print LOGGING.prefix, "Limb.intercept", iter, max_abs_dt

            if (max_abs_dt <= SURFACE_PHOTONS.dlt_precision or
                max_abs_dt >= prev_max_abs_dt): break

        t = t.without_derivs()
        pos = obs + t * los

        if groundtrack:
            track = self.ground.intercept_normal_to(pos, derivs=derivs,
                                                    guess=ground_guess)[0]
            return (pos, t, track)
        else:
            return (pos, t)

    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaulate the surface; ignored.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        return self.ground.normal(pos, derivs=derivs)

################################################################################
# (z,clock) conversions
################################################################################

    def clock_from_groundtrack(self, track, obs, derivs=False):
        """The angle measured clockwise from the projected pole to the
        groundtrack.
        """

        track = Vector3.as_vector3(track, derivs)
        obs = Vector3.as_vector3(obs, derivs)

        xaxis = Vector3.ZAXIS.perp(obs).unit()
        yaxis = Vector3.ZAXIS.ucross(obs).unit()

        x = track.dot(xaxis)
        y = track.dot(yaxis)

        clock = y.arctan2(x) % TWOPI
        return clock

    def groundtrack_from_clock(self, clock, obs, derivs=False):
        """Return the ground point defined by the clock angle."""

        if derivs:
            raise NotImplementedError("Limb.groundtrack_from_clock() " +
                                      "does not implement derivatives")

        clock = Scalar.as_scalar(clock, False)

        obs = Vector3.as_vector3(obs, False)
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs).unit()

        # Groundtrack must fall on the plane defined by these two axes
        a1 = clock.cos() * x_axis + clock.sin() * y_axis
        a2 = obs.unit()

        # Let location of limb be u * a1 + v * a2
        # Unsquash axes...
        b1 = a1.element_mul(self.ground.unsquash)
        b2 = b1.element_mul(self.ground.unsquash)

        # The values of (u,v) must satisfy:
        #   u^2 [b1 dot b1] + u [2v(b1 dot b2)] + [v^2(b2 dot b2) - r_eq^2] = 0
        #
        # Solve for u as a function of v
        #   b1^2 = [b1 dot b1]
        #   b2^2 = [b2 dot b2]
        #
        # u = [-2v[b1 dot b2]
        #      + sqrt(4v^2 [b1 dot b2]^2 - 4 b1^2 [v^2 b2^2 - r_eq^2])] / 2 b1^2
        #
        # u = sqrt(v^2 [(b1 dot b2)^2 - b1^2 b2^2]/b1^4 + [r_eq^2]/b1^2)
        #     - v [b1 dot b2]/b1^2
        #
        # u = sqrt(aa v^2 + bb) - cc v
        # where...
        # aa = [(b1 dot b2)^2 - b1^2 b2^2]/b1^4
        # bb = r_eq^2/b1^2
        # cc = (b1 dot b2)/b1^2

        b1_sq = b1.norm_sq()
        b2_sq = b2.norm_sq()
        b12   = b1.dot(b2)

        aa = (b12**2 - b1_sq * b2_sq) / b1_sq**2
        bb = Scalar(self.ground.req_sq) / b1_sq
        cc = b12 / b1_sq

        # Solve for v via Newton's Method:
        #   u = sqrt(aa v^2 + bb) - cc v
        #   track = u * a1 + v * a2
        #   los = track - obs
        #   normal(track) dot los = 0
        #
        # Define f(v) = normal(track(v)) dot (track(v) - obs)
        # Initial guess is v = 0.

        v = Scalar(np.zeros(clock.shape))
        dv_dv = Scalar(np.ones(clock.shape))

        prev_max_abs_dv = 1.e99

        MAX_ITERS = 8
        DV_CUTOFF = 3.e-16 * self.ground.radii[2]
        for iter in range(8):

            v.insert_deriv('v', dv_dv, override=True)
            u = (aa * v**2 + bb).sqrt() - cc * v
            track = u * a1 + v * a2
            perp = self.ground.normal(track, derivs=True).unit()
            los = (track - obs).unit()
            f = perp.dot(los)

            df_dv = f.d_dv

            dv = f.without_derivs() / df_dv
            v -= dv

            max_abs_dv = abs(dv).max()

            if LOGGING.surface_iterations or Limb.DEBUG:
                print LOGGING.prefix, "Limb.groundtrack_from_clock",
                print iter, max_abs_dv

            if max_abs_dv <= DV_CUTOFF: break
            prev_max_abs_dv = max_abs_dv

        track = u.without_derivs() * a1 + v.without_derivs() * a2
        return track

    def z_clock_from_intercept(self, cept, obs, derivs=False, guess=None,
                                     groundtrack=False):
        """Return z and clock values at an intercept point. """

        if derivs:
            raise NotImplementedError("Limb.z_clock_from_intercept() " +
                                      "does not implement derivatives")

        cept = Vector3.as_vector3(cept, False)
        obs  = Vector3.as_vector3(obs,  False)

        if guess is None:
            ground_guess = False
        else:
            ground_guess = guess

        (track, p) = self.ground.intercept_normal_to(cept, guess=ground_guess)
        z = (cept.norm() - track.norm()).sign() * (cept - track).norm()

        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs).unit()

        x = track.dot(x_axis)
        y = track.dot(y_axis)
        clock = y.arctan2(x) % TWOPI

        results = (z, clock)

        if guess is not None:
            results = (z, clock, p)

        if groundtrack:
            results += (track,)

        return results

    def intercept_from_z_clock(self, z, clock, obs, derivs=False,
                                     groundtrack=False):
        """Return the intercept point defined by z and clock."""

        if derivs:
            raise NotImplementedError("Limb.intercept_from_z_clock() " +
                                      "does not implement derivatives")

        z = Scalar.as_scalar(z, False)
        z = z.mask_where(z.mask, replace=0.)
        clock = Scalar.as_scalar(clock, False)

        obs = Vector3.as_vector3(obs, False)
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs).unit()

        # Groundtrack must fall on the plane defined by these two axes
        a1 = clock.cos() * x_axis + clock.sin() * y_axis
        a2 = obs.unit()

        # Let location of ground point be u * a1 + v * a2
        # Unsquash axes...
        b1 = a1.element_mul(self.ground.unsquash)
        b2 = b1.element_mul(self.ground.unsquash)

        # The values of (u,v) at the ground point must satisfy:
        #   u^2 [b1 dot b1] + u [2v(b1 dot b2)] + [v^2(b2 dot b2) - r_eq^2] = 0
        #
        # Solve for u as a function of v
        #   b1^2 = [b1 dot b1]
        #   b2^2 = [b2 dot b2]
        #
        #
        # u = [-2v[b1 dot b2]
        #      + sqrt(4v^2 [b1 dot b2]^2 - 4 b1^2 [v^2 b2^2 - r_eq^2])] / 2 b1^2
        #
        # u = sqrt(v^2 [(b1 dot b2)^2 - b1^2 b2^2]/b1^4 + [r_eq^2]/b1^2)
        #     - v [b1 dot b2]/b1^2
        #
        # u = sqrt(aa v^2 + bb) - cc v
        # where...
        # aa = [(b1 dot b2)^2 - b1^2 b2^2]/b1^4
        # bb = r_eq^2/b1^2
        # cc = (b1 dot b2)/b1^2

        b1_sq = b1.norm_sq()
        b2_sq = b2.norm_sq()
        b12   = b1.dot(b2)

        aa = (b12**2 - b1_sq * b2_sq) / b1_sq**2
        bb = Scalar(self.ground.req_sq) / b1_sq
        cc = b12 / b1_sq

        # Solve for v via Newton's Method:
        #   u = sqrt(aa v^2 + bb) - cc v
        #   track = u * a1 + v * a2
        #   normal = normal(track).unit()
        #   cept = track + z + normal
        #   los = cept - obs
        #   normal dot los = 0
        #
        # Define f(v) = normal(track(v)) dot (cept(v) - obs)
        # Initial guess is v = 0.

        v = Scalar(np.zeros(clock.shape))
        dv_dv = Scalar(np.ones(clock.shape))

        prev_max_abs_dv = 1.e99

        MAX_ITERS = 10
        DV_CUTOFF = 3.e-16 * self.ground.radii[2]
        for iter in range(MAX_ITERS):

            v.insert_deriv('v', dv_dv, override=True)
            u = (aa * v**2 + bb).sqrt() - cc * v
            track = u * a1 + v * a2
            perp = self.ground.normal(track, derivs=True).unit()
            cept = track + z * perp
            los = (cept - obs).unit()
            f = perp.dot(los)

            df_dv = f.d_dv

            dv = f.without_derivs() / df_dv
            v -= dv

            max_abs_dv = abs(dv).max()

            if LOGGING.surface_iterations or Limb.DEBUG:
                print LOGGING.prefix, "Limb.groundtrack_from_clock",
                print iter, max_abs_dv

            if max_abs_dv <= DV_CUTOFF: break
            prev_max_abs_dv = max_abs_dv

        track = u.without_derivs() * a1 + v.without_derivs() * a2
        cept = track + z * perp.without_derivs()

        if groundtrack:
            return (cept, track)
        else:
            return cept

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric."""

        return self.ground.lon_to_centric(lon, derivs)

    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates."""

        return self.ground.lon_from_centric(lon, derivs)

    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic."""

        return self.ground.lon_to_graphic(lon, derivs)

    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates."""

        return self.ground.lon_from_graphic(lon, derivs)

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetocentric.
        """

        return self.ground.lat_to_centric(lat, lon, derivs)

    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to internal ellipsoid latitude.
        """

        return self.ground.lat_from_centric(lat, lon, derivs)

    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetographic.
        """

        return self.ground.lat_to_graphic(lat, lon, derivs)

    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert planetographic latitude to internal ellipsoid latitude.
        """

        return self.ground.lat_from_graphic(lat, lon, derivs)

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
        from oops.surface_.centricspheroid import CentricSpheroid
        from oops.surface_.graphicspheroid import GraphicSpheroid
        from oops.surface_.centricellipsoid import CentricEllipsoid
        from oops.surface_.graphicellipsoid import GraphicEllipsoid

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.

        NPTS = 1000

        ground = Spheroid("SSB", "J2000", (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

        # Check (z,clock) conversions
        (z, clock, track2) = limb.z_clock_from_intercept(cept, obs, groundtrack=True)

        self.assertTrue((track2 - track).norm().median() < 1.e-10)
        self.assertTrue(abs(track.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)
        self.assertTrue(abs(track2.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)

        matrix = Matrix3.twovec(-obs, 2, Vector3.ZAXIS, 0)
        rotated = matrix * track
        (x,y,_) = (matrix * track).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % TWOPI - clock).median() < 1.e-10)

        (x,y,_) = (matrix * track2).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % TWOPI - clock).max() < 1.e-12)

        self.assertTrue(abs((cept - track).sep(los)  - HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(los) - HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track).sep(limb.ground.normal(track))).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(limb.ground.normal(track2))).median() < 1.e-12)

        cept2 = limb.intercept_from_z_clock(z, clock, obs)
        (z2, clock2) = limb.z_clock_from_intercept(cept2, obs)

        # Validate solution
        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track).unit()
        self.assertTrue(abs(normal.sep(los) - HALFPI).max() < 1.e-12)

        normal2 = cept - track
        sep = (normal2.sep(normal) + HALFPI) % PI - HALFPI
        self.assertTrue(abs(sep).max() < 1.e-10)

        # Validate (lon,lat) conversions
        lon = np.random.random(NPTS) * TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        self.assertTrue(abs(coords[0] - lon).max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        clock = np.random.random(NPTS) * TWOPI
        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        # Validate clock angles
        track = limb.groundtrack_from_clock(clock, obs)
        clock2 = limb.clock_from_groundtrack(track, obs)
        track2 = limb.groundtrack_from_clock(clock2, obs)

        self.assertTrue((track2 - track).norm().max() < 1.e-6)

        dclock = (clock2 - clock + PI) % TWOPI - PI
        self.assertTrue(abs(dclock).max() < 1.e-12)

        ####################

        ground = Ellipsoid("SSB", "J2000", (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        # Check (z,clock) conversions
        (z, clock, track2) = limb.z_clock_from_intercept(cept, obs, groundtrack=True)

        self.assertTrue((track2 - track).norm().median() < 1.e-10)
        self.assertTrue(abs(track.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)
        self.assertTrue(abs(track2.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)

        matrix = Matrix3.twovec(-obs, 2, Vector3.ZAXIS, 0)
        rotated = matrix * track
        (x,y,_) = (matrix * track).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % TWOPI - clock).median() < 1.e-10)

        (x,y,_) = (matrix * track2).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % TWOPI - clock).max() < 1.e-12)

        self.assertTrue(abs((cept - track).sep(los)  - HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(los) - HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track).sep(limb.ground.normal(track))).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(limb.ground.normal(track2))).median() < 1.e-12)

        cept2 = limb.intercept_from_z_clock(z, clock, obs)
        (z2, clock2) = limb.z_clock_from_intercept(cept2, obs)

        # Validate solution
        self.assertTrue(abs(normal.sep(los) - HALFPI).max() < 1.e-12)

        normal2 = cept - track
        sep = (normal2.sep(normal) + HALFPI) % PI - HALFPI
        self.assertTrue(abs(sep).max() < 1.e-10)

        # Validate (lon,lat) conversions
        lon = np.random.random(NPTS) * TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        self.assertTrue(abs(coords[0] - lon).max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        clock = np.random.random(NPTS) * TWOPI
        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        # Validate clock angles
        track = limb.groundtrack_from_clock(clock, obs)
        clock2 = limb.clock_from_groundtrack(track, obs)
        track2 = limb.groundtrack_from_clock(clock2, obs)

        self.assertTrue((track2 - track).norm().max() < 1.e-6)

        dclock = (clock2 - clock + PI) % TWOPI - PI
        self.assertTrue(abs(dclock).max() < 1.e-12)

        ####################

        ground = CentricSpheroid("SSB", "J2000", (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + PI) % TWOPI - PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        ####################

        ground = GraphicSpheroid("SSB", "J2000", (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + PI) % TWOPI - PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        ####################

        ground = CentricEllipsoid("SSB", "J2000", (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + PI) % TWOPI - PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        ####################

        ground = GraphicEllipsoid("SSB", "J2000", (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + PI) % TWOPI - PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
