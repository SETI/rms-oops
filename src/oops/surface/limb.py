##########################################################################################
# oops/surface/limb.py
##########################################################################################

import numpy as np

from polymath import Scalar, Vector3
from oops.config import SURFACE_PHOTONS, LOGGING
from oops.constants import HALFPI, TWOPI
from oops.surface.surface_ import Surface


class Limb(Surface):
    """The locus of points where a surface normal from a spheroid or ellipsoid is
    perpendicular to the line of sight.

    This provides a convenient coordinate system for describing cloud features on the limb
    of a body.

    The coordinates of Limb are `(lon, lat, z)`, much the same as for the surface of the
    associated spheroid or ellipsoid; the difference is in how the intercept point is
    derived.

    * `lon` (Scalar): Longitude at the ground point beneath the limb point, using the same
      definition as that of the associated spheroid or ellipsoid.
    * `lat` (Scalar): Latitude at the ground point beneath the limb point, using the same
      definition as that of the associated spheroid or ellipsoid.
    * `z` (Scalar): The elevation above the surface, as an actual distance measured normal
      to the surface.
    """

    COORDINATE_TYPE = 'limb'
    COORDINATE_NAMES = ('longitude', 'latitude', 'elevation')
    COORDINATE_ABBREVS = ('lon', 'lat', 'z')
    COORDINATE_RANGES = ((0, TWOPI), (-HALFPI, HALFPI), (None, None))
    IS_VIRTUAL = True

    _DEBUG = False          # Set to True for convergence testing

    def __init__(self, ground, *, limits=None):
        """Constructor for a Limb surface.

        Parameters:
            ground (Surface): Object relative to which limb points are to be defined. It
                should be a Spheroid or Ellipsoid, optionally using Centric or Graphic
                coordinates.
            limits (tuple[float, float], optional): A pair of values defining the lower
                and upper limits placed on `z`; values outside this range are masked.
        """

        if ground.COORDINATE_TYPE != 'spherical':
            raise ValueError('Limb requires an ellipsoidal ground surface')

        self._ground = ground
        self.origin = ground.origin
        self.frame  = ground.frame

        if limits is None:
            self._limits = None
        else:
            self._limits = (limits[0], limits[1])

        # Save the unmasked version of this surface. It must be of this class, not
        # necessarily Limb; a PolarLimb defines different coordinates under the same
        # method names.
        if limits is None:
            self.unmasked = self
        else:
            self.unmasked = type(self)(self._ground)

        # Unique key for intercept calculations
        self.intercept_key = ('limb',) + self._ground.intercept_key

    @property
    def ground(self):
        """The surface relative to which this limb is defined.

        Returns:
            Surface: The ground surface, typically an Ellipsoid or Spheroid.
        """

        return self._ground

    @property
    def limits(self):
        """The range of vertical distances from the ground surface, in km.

        Returns:
            tuple or None: The (lower, upper) limits in km, or None if unlimited.
        """

        return self._limits

    def __getstate__(self):
        self.refresh()
        return (self._ground, self._limits)

    def __setstate__(self, state):
        (ground, limits) = state
        self.__init__(ground, limits=limits)
        self.freeze()

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (lon, lat) or all three (lon, lat, z) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Scalar, optional): Optionally, the value of the coefficient `p` such
                that `ground + p * normal(ground) = pos`, for the ground point associated
                with the position. If it is not None, the converged value of `p` is
                appended to the returned tuple; use `hints=True` if you lack an initial
                value but require the new value to be returned.
            groundtrack (bool, optional): True to append the intercept point on the
                surface to the returned tuple.

        Returns:
            tuple: Two to five values, where:

            * `lon` (Scalar): Longitude in radians.
            * `lat` (Scalar): Latitude in radians.
            * `z` (Scalar): Vertical altitude in km normal to the body surface; included
              if axes == 3.
            * `p` (Scalar): The converged coefficient; included if the input value of
              `hints` is not None.
            * `track`: Associated point on the body surface; included if the input
              groundtrack is True.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # There's a quick solution for the ground point if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            (track, p) = self._ground.intercept_normal_to(pos, derivs=derivs,
                                                         guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self._ground._unsquash_sq
            track = pos.element_div(denom)

        (lon, lat) = self._ground.coords_from_vector3(track, derivs=derivs)

        # Derive z; mask if necessary
        if axes == 3 or self._limits is not None:
            z = (pos - track).norm() * p.sign()

            if self._limits is not None:
                zmask = z.tvl_lt(self._limits[0]) | z.tvl_gt(self._limits[1])
                if zmask.any():
                    z = z.remask_or(zmask)
                    lon = lon.remask(z.mask)
                    lat = lat.remask(z.mask)

        results = (lon, lat)

        if axes == 3:
            results += (z,)

        if hints is not None:
            results += (p,)

        if groundtrack:
            results += (track,)

        return results

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None, groundtrack=False):
        """The position where a point with the given coordinates falls relative to this
        surface's origin and frame.

        Parameters:
            coords (tuple[Scalar, ...]): Two or three Scalars defining coordinates at
                or near this surface. These can have different shapes, but must be
                broadcastable to a common shape.

                * `lon` (rad): Longitude.
                * `lat` (rad): Latitude.
                * `z` (km, optional): Perpendicular distance from the limb surface.

            obs (Vector3, optional): Observer positions relative to this Surface's origin
                and frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to include the partial derivatives of the
                intercept point with respect to observer and to the coordinates.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.
            groundtrack (bool, optional): True to append the associated groundtrack points
                on the body surface to the returned result.

        Returns:
            Vector3 or tuple: `pos` or `(pos[, hints][, track])`, where:

            * `pos` (Vector3): Points defined by the coordinates, relative to this
              surface's origin and frame.
            * `hints` (Any): The input value of `hints`, included if it is not None.
            * `track` (Vector3): Associated points on the body surface; included if
              `groundtrack` is True.
        """

        pos = self._ground.vector3_from_coords(coords, derivs=derivs)

        results = (pos,)
        if hints is not None:
            results += (hints,)

        if groundtrack:
            results += (self._ground.vector3_from_coords(coords[:2], derivs=derivs),)

        if len(results) == 1:
            return pos

        return results

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False,
                  guess=None, hints=None, groundtrack=False):
        """The position where a specified line of sight intercepts the Surface.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this Surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this Surface's frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored here.
            derivs (bool, optional): True to propagate any derivatives inside obs and los
                into the returned intercept point.
            guess (Scalar, optional): Unused.
            hints (Scalar, optional): Optional initial guess at the coefficient `p` such
                that `ground + p * normal(ground) = limb_intercept`, for the ground point
                on the body surface associated with the limb intercept point being sought.
                If it is not None, the converged value is appended to the returned tuple;
                use `hints=True` if you lack an initial guess but require the converged
                value of `p` to be returned.
            groundtrack (bool, optional): True to append the associated body surface
                points to the returned results.

        Returns:
            tuple: Two to four values, where:

            * `pos` (Vector3): Intercept points on the Surface relative to this surface's
              origin and frame, in km.
            * `t` (Scalar): Value such that `intercept = obs + t * los`.
            * `p` (Scalar): The converged solution such that::

                ground + p * normal(ground) = limb_intercept;

              included if the input value of `hints` is not None.
            * `track` (Vector3): Groundtrack points on the body surface; included if the
              input value of `groundtrack` is True.
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

        # The precision of t should match the default geometric accuracy defined by
        # SURFACE_PHOTONS.km_precision. Set our precision goal on t accordingly.
        km_scale = los.norm().max().vals
        precision = SURFACE_PHOTONS.km_precision / km_scale

        max_abs_dt = 1.e99
        converged = False
        for count in range(SURFACE_PHOTONS.max_iterations):
            pos = obs + t * los
            pos.insert_deriv('_pos_', Vector3.IDENTITY)

            (track,
             ground_guess) = self._ground.intercept_normal_to(pos, derivs=True,
                                                             guess=ground_guess)
            normal = self._ground.normal(track, derivs=True)

            f = normal.dot(los)
            df_dt = normal.d_d_pos_.chain(los).dot(los)
            dt = f / df_dt
            t = t - dt.without_deriv('_pos_')

            prev_max_abs_dt = max_abs_dt
            max_abs_dt = abs(dt).max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Limb._DEBUG:
                LOGGING.convergence(f'{type(self).__name__}.intercept: iter={count+1}; '
                                    f'change[km]={max_abs_dt*km_scale:.6g}')

            if max_abs_dt <= precision:
                converged = True
                break

            if max_abs_dt >= prev_max_abs_dt:
                break

        if not converged:
            LOGGING.warn(f'{type(self).__name__}.intercept did not converge: '
                         f'iter={count+1}; change[km]={max_abs_dt*km_scale:.6g}')

        # Make sure all values are consistent with t
        pos = obs + t * los
        results = (pos, t)

        if hints is not None or groundtrack:
            (track,
             ground_guess) = self._ground.intercept_normal_to(pos, derivs=True,
                                                             guess=ground_guess)

            if hints is not None:
                results = results + (ground_guess,)

            if groundtrack:
                results = results + (track,)

        return results

    def normal(self, pos, *, obs=None, time=None, derivs=False, hints=None):
        """The normal vector at a position at or near a surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives of pos into the
                returned normal vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Directions normal to the Surface that pass
            through the position, optionally followed by `hints`. Vector lengths are
            arbitrary, and the input value of `hints` is returned if it is not None.
        """

        return self._ground.normal(pos, obs=obs, time=time, derivs=derivs, hints=hints)

    ######################################################################################
    # (z,clock) conversions
    ######################################################################################

    def clock_from_groundtrack(self, track, obs, *, derivs=False, hints=None):
        """The angle measured clockwise from the projected pole to the groundtrack's
        surface normal.

        Parameters:
            track (Vector3): Positions at or near the ellipsoid's surface relative to the
                ellipsoid's origin and frame.
            obs (Vector3): Observer positions relative to the ellipsoid's origin and
                frame.
            derivs (bool, optional): True to propagate derivatives of track and obs into
                the returned clock angle.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this method. If it is not None, its value
                is appended to the returned tuple.

        Returns:
            Scalar or tuple[Scalar, Any]: The clock angle in radians, measured clockwise
            from the projected pole, optionally followed by `hints`. The input value of
            `hints` is returned if it is not None.
        """

        track = Vector3.as_vector3(track, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # Get groundtrack surface normal
        normal = self._ground.normal(track, derivs=derivs)

        # Define the axes of the "clock"
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)

        # Derive the angle
        normal_x = normal.dot(x_axis)
        normal_y = normal.dot(y_axis)

        clock = normal_y.arctan2(normal_x) % Scalar.TWOPI

        if hints is not None:
            return (clock, hints)

        return clock

    def groundtrack_from_clock(self, clock, obs, *, derivs=False, hints=None):
        """The ground point defined by the clock angle and observation point.

        Parameters:
            clock (Scalar): Angle of the ellipsoid's normal vector, measured clockwise
                from the projected pole.
            obs (Vector3): Observer positions relative to the ellipsoid's origin and
                frame.
            derivs (bool, optional): True to propagate derivatives of clock and obs into
                the returned ellipsoid surface point.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this method. If it is not None, its value
                is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: The ground point on the ellipsoid, optionally
            followed by `hints`. The input value of `hints` is returned if it is not None.
        """

        clock = Scalar.as_scalar(clock, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # Define the required direction of the surface normal
        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)
        normal = clock.cos() * x_axis + clock.sin() * y_axis

        return self._ground.intercept_with_normal(normal, derivs=derivs, hints=hints)

    def z_clock_from_intercept(self, pos, obs, *, derivs=False, hints=None,
                               groundtrack=False):
        """The z and clock values at a limb intercept point.

        Parameters:
            pos (Vector3): Limb intercept points relative to this Surface's origin and
                frame.
            obs (Vector3): Observer positions relative to the ellipsoid's origin and
                frame.
            derivs (bool, optional): True to propagate derivatives of pos and obs into the
                returned values.
            hints (Scalar, optional): Optional value of the coefficient `p` such that
                `ground + p * normal(ground) = pos`. If it is not None, the value of `p`
                is appended to the returned tuple; use `hints=True` if you lack an initial
                value but require the new value to be returned.
            groundtrack (bool, optional): True to append the associated groundtrack points
                on the body surface to the returned tuple.

        Returns:
            tuple: Two to four values, where:

            * `z` (Scalar): The perpendicular distance from the ellipsoidal surface, in
              km.
            * `clock` (Scalar): Angle of the ellipsoid's normal vector, measured clockwise
              from the projected pole.
            * `p` (Scalar): The coefficient described above; included if the input value
              of `hints` is not None.
            * `track` (Vector3): Groundtrack points on the body surface; included if
              `groundtrack` is True.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # There's a quick solution for the surface point if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            if hints is None:
                track = self._ground.intercept_normal_to(pos, derivs=derivs)
                p = None
            else:
                (track, p) = self._ground.intercept_normal_to(pos, derivs=derivs,
                                                             guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self._ground._unsquash_sq
            track = pos.element_div(denom)

        normal = self._ground.normal(track, derivs=derivs)

        z = normal.unit().dot(pos - track)

        x_axis = Vector3.ZAXIS.perp(obs).unit()
        y_axis = Vector3.ZAXIS.ucross(obs)

        x = normal.dot(x_axis)
        y = normal.dot(y_axis)
        clock = y.arctan2(x) % Scalar.TWOPI

        results = (z, clock)

        if hints is not None:
            results += (p,)

        if groundtrack:
            results += (track,)

        return results

    def intercept_from_z_clock(self, z, clock, obs, *, derivs=False, hints=None,
                               groundtrack=False):
        """The limb intercept point as defined by z and clock.

        Parameters:
            z (Scalar): The perpendicular distance in km from the body surface.
            clock (Scalar): Angle of the ellipsoid's normal vector, measured clockwise
                from the projected pole.
            obs (Vector3): Observer positions relative to the ellipsoid's origin and
                frame.
            derivs (bool, optional): True to propagate derivatives of z, clock, and obs
                into the returned limb intercept points.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this method. If it is not None, its value
                is appended to the returned tuple.
            groundtrack (bool, optional): True to append the associated points on the
                surface of the ellipsoid to the returned tuple.

        Returns:
            Vector3 or tuple: `intercept` or `(intercept[, hints][, track])`, where:

            * `intercept` (Vector3): Limb surface intercept points.
            * `hints` (Any): The input value of `hints`, included if it is not None.
            * `track` (Vector3): The associated points on the surface of the ellipsoid;
              included if `groundtrack` is True.
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
        #   surface(normal) = normal.element_mul(self._ground._squash)
        #                           .with_norm(self._ground._req)
        #                           .element_mul(self._ground._squash)
        #
        # Substituting the first equation into the second,
        #   normal dot surface(normal) + z - normal dot obs = 0
        #
        # Now we can solve for p using Newton's Method.
        #   f(p) = normal dot (surface(normal) - obs) + z = 0

        # Make an initial guess at p
        axis1_unsq = axis1.wod.element_mul(self._ground._unsquash)
        req = self._ground._req / axis1_unsq.norm()
            # This is the approximate body radius on axis1
        p = ((req + z.wod) / obs.wod.norm()).arcsin()

        # Iterate until convergence stops
        max_dp = 1.e99
        converged = False

        # Extra steps are often needed for convergence
        for count in range(SURFACE_PHOTONS.max_iterations + 10):

            p.insert_deriv('_p_', Scalar.ONE)
            normal = p.cos() * axis1 + p.sin() * axis2
            s1 = normal.element_mul(self._ground._squash)
            s2 = s1.with_norm(self._ground._req)
            surface = s2.element_mul(self._ground._squash)

            # The solution is undefined if obs is closer than z!
            mask = ((obs - surface).norm() <= z).vals | surface.mask

            # One step of Newton's method
            f = normal.dot(surface - obs) + z
            dp = f.without_deriv('_p_') / f.d_d_p_
            dp[mask] = 0
            p -= dp

            prev_max_dp = max_dp
            max_dp = dp.abs().max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Limb._DEBUG:
                LOGGING.convergence('%s.intercept_from_z_clock(): '
                                    'iter=%d; change=%.6g'
                                    % (type(self).__name__, count+1, max_dp))

            if max_dp <= SURFACE_PHOTONS.rel_precision:
                converged = True
                break

            if max_dp >= prev_max_dp:
                break

        if not converged:
            LOGGING.warn('%s.intercept_from_z_clock() did not converge: '
                         'iter=%d; change=%.6g'
                         % (type(self).__name__, count+1, max_dp))

        p = p.without_deriv('_p_')
        normal = p.cos() * axis1 + p.sin() * axis2
        s1 = normal.element_mul(self._ground._squash)
        s2 = s1.with_norm(self._ground._req)
        surface = s2.element_mul(self._ground._squash)
        pos = surface + z * normal

        results = (pos,)

        if hints is not None:
            results += (hints,)

        if groundtrack:
            results += (surface,)

        if len(results) == 1:
            return pos

        return results

    ######################################################################################
    # Longitude conversions
    ######################################################################################

    def lon_to_centric(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetocentric longitude.
        """

        return self._ground.lon_to_centric(lon, derivs=derivs)

    def lon_from_centric(self, lon, *, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        return self._ground.lon_from_centric(lon, derivs=derivs)

    def lon_to_graphic(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetographic longitude.
        """

        return self._ground.lon_to_graphic(lon, derivs=derivs)

    def lon_from_graphic(self, lon, *, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        return self._ground.lon_from_graphic(lon, derivs=derivs)

    ######################################################################################
    # Latitude conversions
    ######################################################################################

    def lat_to_centric(self, lat, lon, *, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetocentric.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Planetocentric latitude.
        """

        return self._ground.lat_to_centric(lat, lon, derivs=derivs)

    def lat_from_centric(self, lat, lon, *, derivs=False):
        """Convert planetocentric latitude to internal ellipsoid latitude.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Squashed latitude.
        """

        return self._ground.lat_from_centric(lat, lon, derivs=derivs)

    def lat_to_graphic(self, lat, lon, *, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetographic.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Planetographic latitude.
        """

        return self._ground.lat_to_graphic(lat, lon, derivs=derivs)

    def lat_from_graphic(self, lat, lon, *, derivs=False):
        """Convert a planetographic latitude to internal ellipsoid latitude.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Squashed latitude.
        """

        return self._ground.lat_from_graphic(lat, lon, derivs=derivs)

    ######################################################################################
    # (lon,lat) conversions
    ######################################################################################

    def lonlat_from_vector3(self, pos, *, derivs=False, hints=None, groundtrack=True):
        """Longitude and latitude for a position near the surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            derivs (bool, optional): True to propagate derivatives of pos into the
                returned coordinates.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this method. If it is not None, its value
                is appended to the returned tuple.
            groundtrack (bool, optional): True to append the associated groundtrack points
                on the body surface to the returned tuple.

        Returns:
            tuple: Two to four values, where:

            * `lon` (Scalar): Longitude at the surface in radians.
            * `lat` (Scalar): Latitude at the surface in radians.
            * `hints` (Any): The input value of `hints`, included if it is not None.
            * `track` (Vector3): Groundtrack points on the body surface; included if
              `groundtrack` is True.
        """

        track = self._ground.intercept_normal_to(pos, derivs=derivs)
        coords = self._ground.coords_from_vector3(track, derivs=derivs)

        results = (coords[0], coords[1])

        if hints is not None:
            results += (hints,)

        if groundtrack:
            results += (track,)

        return results

##########################################################################################
