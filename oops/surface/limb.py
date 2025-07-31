################################################################################
# oops/surface/limb.py: Limb subclass of class Surface
################################################################################

import numpy as np

from polymath              import Scalar, Vector3
from oops.config           import SURFACE_PHOTONS, LOGGING
from oops.surface.surface_ import Surface

class Limb(Surface):
    """This surface is defined as the locus of points where a surface normal
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
                normal to the surface.
    """

    COORDINATE_TYPE = 'limb'
    IS_VIRTUAL = True
    DEBUG = False           # True for convergence testing

    #===========================================================================
    def __init__(self, ground, limits=None):
        """Constructor for a Limb surface.

        Input:
            ground      the Surface object relative to which limb points are to
                        be defined. It should be a Spheroid or Ellipsoid,
                        optically using Centric or Graphic coordinates.
            limits      an optional single value or tuple defining the absolute
                        numerical limit(s) placed on z; values outside this
                        range are masked.
        """

        if ground.COORDINATE_TYPE != 'spherical':
            raise ValueError('Limb requires an ellipsoidal ground surface')

        self.ground = ground
        self.origin = ground.origin
        self.frame  = ground.frame

        if limits is None:
            self.limits = None
        else:
            self.limits = (limits[0], limits[1])

        # Save the unmasked version of this surface
        if limits is None:
            self.unmasked = self
        else:
            self.unmasked = Limb(self.ground, None)

        # Unique key for intercept calculations
        self.intercept_key = ('limb',) + self.ground.intercept_key

    def __getstate__(self):
        return (self.ground, self.limits)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Input:
            pos         a Vector3 of positions at or near the surface, relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            axes        2 or 3, indicating whether to return the first two
                        coordinates (lon, lat) or all three (lon, lat, z) as
                        Scalars.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       optionally, the value of the coefficient p such that:
                            ground + p * normal(ground) = pos;
                        for the ground point associate with the position.
                        Ignored if the value is None (the default) or True.
            groundtrack True to return the intercept on the surface along with
                        the coordinates.

        Return:         a tuple containing two to four values.
            lon         longitude in radians.
            lat         latitude in radians.
            z           vertical altitude in km normal to the body surface;
                        included if axes == 3.
            track       associated point on the body surface; included if the
                        input groundtrack is True.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # There's a quick solution for the ground point if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            (track, p) = self.ground.intercept_normal_to(pos, derivs=derivs,
                                                         guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self.ground.unsquash_sq
            track = pos.element_div(denom)

        (lon, lat) = self.ground.coords_from_vector3(track, derivs=derivs)

        # Derive z; mask if necessary
        if axes == 3 or self.limits is not None:
            z = (pos - track).norm() * p.sign()

            if self.limits is not None:
                zmask = z.tvl_lt(self.limits[0]) | z.tvl_gt(self.limits[1])
                if zmask.any():
                    z = z.remask_or(zmask)
                    lon = lon.remask(z.mask)
                    lat = lat.remask(z.mask)

        results = (lon, lat)

        if axes == 3:
            results += (z,)

        if groundtrack:
            results += (track,)

        return results

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False,
                                          groundtrack=False):
        """The position where a point with the given coordinates falls relative
        to this surface's origin and frame.

        Input:
            coords      a tuple of two or three Scalars defining coordinates at
                        or near this surface. These can have different shapes,
                        but must be broadcastable to a common shape.
                lon     longitude in radians.
                lat     latitude in radians.
                z       the perpendicular distance in km from the limb surface.
            obs         a Vector3 of observer positions relative to this
                        surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.
            groundtrack True to include the associated groundtrack points on the
                        body surface in the returned result.

        Return:         pos or (pos, track), where:
            pos         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
            track       a Vector3 of associated points on the body surface;
                        included if input groundtrack is True.
        """

        pos = self.ground.vector3_from_coords(coords, derivs=derivs)

        if not groundtrack:
            return pos

        track = self.ground.vector3_from_coords(coords[:2], derivs=derivs)
        return (pos, track)

    #===========================================================================
    def intercept(self, obs, los, time=None, direction='dep', derivs=False,
                                  guess=None, hints=None, groundtrack=False):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3 relative to this
                        surface's origin and frame.
            los         line of sight as a Vector3 in this surface's frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface; ignored here.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       unused.
            hints       optional initial guess a coefficient p such that:
                            ground + p * normal(ground) = limb_intercept
                        for the ground point on the body surface associated with
                        the limb intercept point being sought. The converged
                        value will be included in the tuple returned. Use
                        hints=True if you do not have an initial guess but still
                        would like the converged value of p to be returned.
            groundtrack True to include the associated body surface points in
                        the returned results.

        Return:         a tuple of two to four values.
            pos         a Vector3 of intercept points on the surface relative
                        to this surface's origin and frame, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
            p           the converged solution such that
                            ground + p * normal(ground) = limb_intercept;
                        included if the input value of hints is not None.
            track       the Vector3 of groundtrack points on the body surface;
                        included if the input value of groundtrack is True.
        """

        obs = Vector3.as_vector3(obs, recursive=derivs)
        los = Vector3.as_vector3(los, recursive=derivs)

        # Solve for the intercept distance where the line of sight is normal to
        # the surface.
        #
        # pos = obs + t * los
        # track = ground.intercept_normal_to(pos(t))
        # normal(track) dot los = 0
        #
        # Solve for t.
        #   f(t) = normal(track(pos(t))) dot los
        #
        #   df/dt = dnormal/dt dot los
        #         = (dnormal/dpos <chain> dpos/dt) dot los
        #         = (dnormal/dpos <chain> los) dot los
        #
        # Initial guess is where los and pos are perpendicular:
        # (obs + t * los) dot los = 0
        #
        # t = -(obs dot los) / (los dot los)

        los_wod = los.wod
        t = -obs.wod.dot(los_wod) / los_wod.dot(los_wod)

        # Hints is the value of ground_guess
        if isinstance(hints, (type(None), bool, np.bool_)):
            ground_guess = True
        else:
            ground_guess = hints.wod

        # The precision of t should match the default geometric accuracy defined
        # by SURFACE_PHOTONS.km_precision. Set our precision goal on t
        # accordingly.
        km_scale = los.norm().max().vals
        precision = SURFACE_PHOTONS.km_precision / km_scale

        max_abs_dt = 1.e99
        converged = False
        for count in range(SURFACE_PHOTONS.max_iterations):
            pos = obs + t * los
            pos.insert_deriv('_pos_', Vector3.IDENTITY)

            (track,
             ground_guess) = self.ground.intercept_normal_to(pos, derivs=True,
                                                             guess=ground_guess)
            normal = self.ground.normal(track, derivs=True)

            f = normal.dot(los)
            df_dt = normal.d_d_pos_.chain(los).dot(los)
            dt = f / df_dt
            t = t - dt.without_deriv('_pos_')

            prev_max_abs_dt = max_abs_dt
            max_abs_dt = abs(dt).max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Limb.DEBUG:
                LOGGING.convergence('%s.intercept(): iter=%d; change[km]=%.6g'
                                    % (type(self).__name__, count+1,
                                       max_abs_dt * km_scale))

            if max_abs_dt <= precision:
                converged = True
                break

            if max_abs_dt >= prev_max_abs_dt:
                break

        if not converged:
            LOGGING.warn('%s.intercept() did not converge: '
                         'iter=%d; change[km]=%.6g'
                         % (type(self).__name__, count+1, max_abs_dt*km_scale))

        # Make sure all values are consistent with t
        pos = obs + t * los
        results = (pos, t)

        if hints is not None or groundtrack:
            (track,
             ground_guess) = self.ground.intercept_normal_to(
                                                        pos, derivs=True,
                                                        guess=ground_guess)

            if hints is not None:
                results = results + (ground_guess,)

            if groundtrack:
                results = results + (track,)

        return results

    #===========================================================================
    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        return self.ground.normal(pos, derivs=derivs)

    ############################################################################
    # (z,clock) conversions
    ############################################################################

    def clock_from_groundtrack(self, track, obs, derivs=False):
        """The angle measured clockwise from the projected pole to the
        groundtrack's surface normal.

        Input:
            track       a Vector3 of positions at or near the ellipsoid's
                        surface relative to the ellipsoid's origin and frame.
            obs         a Vector3 of observer positions relative to the
                        ellipsoid's origin and frame.
            derivs      True to propagate derivatives of track and obs into the
                        returned clock angle.
        """

        track = Vector3.as_vector3(track, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # Get groundtrack surface normal
        normal = self.ground.normal(track, derivs=derivs)

        # Define the axes of the "clock"
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)

        # Derive the angle
        normal_x = normal.dot(x_axis)
        normal_y = normal.dot(y_axis)

        clock = normal_y.arctan2(normal_x) % Scalar.TWOPI
        return clock

    #===========================================================================
    def groundtrack_from_clock(self, clock, obs, derivs=False):
        """The ground point defined by the clock angle and observation point.

        Input:
            clock       angle of the ellipsoid's normal vector, measured
                        clockwise from the projected pole.
            obs         a Vector3 of observer positions relative to the
                        ellipsoid's origin and frame.
            derivs      True to propagate derivatives of clock and obs into the
                        returned ellipsoid surface point.
        """

        clock = Scalar.as_scalar(clock, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # Define the required direction of the surface normal
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)
        normal = clock.cos() * x_axis + clock.sin() * y_axis

        return self.ground.intercept_with_normal(normal, derivs=derivs)

    #===========================================================================
    def z_clock_from_intercept(self, pos, obs, derivs=False, hints=None,
                                               groundtrack=False):
        """The z and clock values at a limb intercept point.

        Input:
            pos         a Vector3 of positions at the limb intercept point,
                        relative to this surface's origin and frame.
            obs         a Vector3 of observer positions relative to the
                        ellipsoid's origin and frame.
            derivs      True to propagate derivatives of clock and obs into the
                        returned ellipsoid surface point.
            hints       optionally, the value of the coefficient p such that
                            ground + p * normal(ground) = pos
                        for the ground point on the body surface.
            groundtrack True to include the surface intercept points of the body
                        associated with each limb intercept.

        Return          (z, clock) or (z, clock, track), where
            z           the perpendicular distance from the ellipsoidal surface,
                        in km.
            clock       angle of the ellipsoid's normal vector, measured
                        clockwise from the projected pole.
            track       the Vector3 of groundtrack points on the body surface;
                        included if the input value of groundtrack is True.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # There's a quick solution for the surface point if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            track = self.ground.intercept_normal_to(pos, derivs=derivs)
        else:
            denom = Vector3.ONES + hints * self.ground.unsquash_sq
            track = pos.element_div(denom)

        normal = self.ground.normal(track, derivs=derivs)

        z = pos.norm() - track.norm()

        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)

        x = normal.dot(x_axis)
        y = normal.dot(y_axis)
        clock = y.arctan2(x) % Scalar.TWOPI

        if groundtrack:
            return (z, clock, track)

        return (z, clock)

    #===========================================================================
    def intercept_from_z_clock(self, z, clock, obs, derivs=False,
                                     groundtrack=False):
        """The limb intercept point as defined by z and clock.

        Input:
            z           the perpendicular distance in km from the body surface.
            clock       angle of the ellipsoid's normal vector, measured
                        clockwise from the projected pole.
            obs         a Vector3 of observer positions relative to the
                        ellipsoid's origin and frame.
            derivs      True to propagate derivatives of z, clock, and obs into
                        the returned limb intercept points.
            groundtrack if True, the tuple (limb intercept, ground track) is
                        returned rather than just the limb intercept.

        Return:         intercept or (intercept, track), where
            intercept   Vector3 of limb surface intercept points.
            track       Vector3 of the associated points on the surface of the
                        Ellipsoid; included if groundtrack is True.
        """

        z = Scalar.as_scalar(z, recursive=derivs)
        z = z.mask_where(z.mask, replace=0.)
        clock = Scalar.as_scalar(clock, recursive=derivs)

        obs = Vector3.as_vector3(obs, recursive=derivs)
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)

        # Groundtrack's normal must fall on the plane defined by these two axes
        axis1 = clock.cos() * x_axis + clock.sin() * y_axis
        axis2 = obs.unit()

        # Let the unit normal vector at the ellipsoid surface be
        #   normal = cos(p) * axis1 + sin(p) * axis2
        #
        # We need to solve for limb point "pos" such that
        #   surface(normal) + z * normal = pos
        #   normal dot (pos - obs) = 0
        #
        # where
        #   surface(normal) = normal.element_mul(self.ground.squash)
        #                           .with_norm(self.ground.req)
        #                           .element_mul(self.ground.squash)
        #
        # Substituting the first equation into the second,
        #   normal dot surface(normal) + z - normal dot obs = 0
        #
        # Now we can solve for p using Newton's Method.
        #   f(p) = normal dot (surface(normal) - obs) + z = 0

        # Make an initial guess at p
        axis1_unsq = axis1.wod.element_mul(self.ground.unsquash)
        req = self.ground.req / axis1_unsq.norm()
            # This is the approximate body radius on axis1
        p = ((req + z.wod) / obs.wod.norm()).arcsin()

        # Iterate until convergence stops
        max_dp = 1.e99
        converged = False

        # Extra steps are often needed for convergence
        for count in range(SURFACE_PHOTONS.max_iterations + 10):

            p.insert_deriv('_p_', Scalar.ONE)
            normal = p.cos() * axis1 + p.sin() * axis2
            s1 = normal.element_mul(self.ground.squash)
            s2 = s1.with_norm(self.ground.req)
            surface = s2.element_mul(self.ground.squash)

            # The solution is undefined if obs is closer than z!
            mask = ((obs - surface).norm() <= z).vals | surface.mask

            # One step of Newton's method
            f = normal.dot(surface - obs) + z
            dp = f.without_deriv('_p_') / f.d_d_p_
            dp[mask] = 0
            p -= dp

            max_dp = dp.abs().max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Limb.DEBUG:
                LOGGING.convergence('%s.intercept_from_z_clock(): '
                                    'iter=%d; change=%.6g'
                                    % (type(self).__name__, count+1, max_dp))

            if max_dp <= SURFACE_PHOTONS.rel_precision:
                converged = True
                break

        if not converged:
            LOGGING.warn('%s.intercept_from_z_clock() did not converge: '
                         'iter=%d; change=%.6g'
                         % (type(self).__name__, count+1, max_dp))

        p = p.without_deriv('_p_')
        normal = p.cos() * axis1 + p.sin() * axis2
        s1 = normal.element_mul(self.ground.squash)
        s2 = s1.with_norm(self.ground.req)
        surface = s2.element_mul(self.ground.squash)
        pos = surface + z * normal

        if groundtrack:
            return (pos, surface)
        else:
            return pos

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return self.ground.lon_to_centric(lon, derivs)

    #===========================================================================
    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        return self.ground.lon_from_centric(lon, derivs)

    #===========================================================================
    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        return self.ground.lon_to_graphic(lon, derivs)

    #===========================================================================
    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        return self.ground.lon_from_graphic(lon, derivs)

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetocentric.

        Input:
            lat         squashed latitide, radians.
            lon         squashed longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return self.ground.lat_to_centric(lat, lon, derivs)

    #===========================================================================
    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to internal ellipsoid latitude.

        Input:
            lat         planetocentric latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          squashed latitude.
        """

        return self.ground.lat_from_centric(lat, lon, derivs)

    #===========================================================================
    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetographic.

        Input:
            lat         squashed latitide, radians.
            lon         squashed longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        return self.ground.lat_to_graphic(lat, lon, derivs)

    #===========================================================================
    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert a planetographic latitude to internal ellipsoid latitude.

        Input:
            lat         planetographic latitide, radians.
            lon         planetographic longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          squashed latitude.
        """

        return self.ground.lat_from_graphic(lat, lon, derivs)

    ############################################################################
    # (lon,lat) conversions
    ############################################################################

    def lonlat_from_vector3(self, pos, derivs=False, groundtrack=True):
        """Longitude and latitude for a position near the surface."""

        track = self.ground.intercept_normal_to(pos, derivs=derivs)
        coords = self.ground.coords_from_vector3(track, derivs=derivs)

        if groundtrack:
            return (coords[0], coords[1], track)
        else:
            return coords[:2]

################################################################################
