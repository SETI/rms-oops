################################################################################
# oops/surface/ellipsoid.py: Ellipsoid subclass of class Surface
################################################################################

import numpy as np

from polymath              import Matrix, Scalar, Vector3
from oops.config           import SURFACE_PHOTONS, LOGGING
from oops.frame.frame_     import Frame
from oops.path.path_       import Path
from oops.surface.surface_ import Surface

class Ellipsoid(Surface):
    """An ellipsoidal surface centered on the given path and fixed with respect
    to the given frame. The short radius of the ellipsoid is oriented along the
    Z-axis of the frame and the long radius is along the X-axis.

    The coordinates defining the surface grid are (longitude, latitude).
    Both are based on the assumption that a spherical body has been "squashed"
    along the Y- and Z-axes. The latitudes and longitudes defined in this manner
    are neither planetocentric nor planetographic; functions are provided to
    perform conversions to either choice. Longitudes are measured in a right-
    handed manner, increasing toward the east; values range from 0 to 2*pi.

    The third coordinate is z, which measures vertical distance in km along the
    normal vector from the surface.
    """

    COORDINATE_TYPE = 'spherical'
    IS_VIRTUAL = False
    HAS_INTERIOR = True

    DEBUG = False       # True for convergence testing in intercept_normal_to()

    #===========================================================================
    def __init__(self, origin, frame, radii, exclusion=0.9):
        """Constructor for an Ellipsoid object.

        Input:
            origin      the Path object or ID defining the center of the
                        ellipsoid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the ellipsoid is fixed, with the shortest radius
                        of the ellipsoid along the Z-axis and the longest radius
                        along the X-axis.
            radii       a tuple (a,b,c) containing the radii from longest to
                        shortest, in km.
            exclusion   the fraction of the polar radius within which
                        calculations of intercept_normal_to() are suppressed.
                        Values of less than 0.95 are not recommended because
                        the problem becomes numerically unstable.
        """

        self.origin = Path.as_waypoint(origin)
        self.frame  = Frame.as_wayframe(frame)

        self.radii    = np.asfarray(radii)
        self.radii_sq = self.radii**2
        self.req      = self.radii[0]
        self.req_sq   = self.req**2
        self.rpol     = self.radii[2]

        self.squash_y       = self.radii[1] / self.radii[0]
        self.squash_y_sq    = self.squash_y**2
        self.unsquash_y     = self.radii[0] / self.radii[1]
        self.unsquash_y_sq  = self.unsquash_y**2

        self.squash_z       = self.radii[2] / self.radii[0]
        self.squash_z_sq    = self.squash_z**2
        self.unsquash_z     = self.radii[0] / self.radii[2]
        self.unsquash_z_sq  = self.unsquash_z**2

        self.squash         = Vector3((1., self.squash_y, self.squash_z))
        self.squash_sq      = self.squash.element_mul(self.squash)
        self.unsquash       = Vector3((1., 1./self.squash_y, 1./self.squash_z))
        self.unsquash_sq    = self.unsquash.element_mul(self.unsquash)

        self.unsquash_sq_2d = Matrix(([1.,0.,0.],
                                      [0.,self.unsquash_y**2,0.],
                                      [0.,0.,self.unsquash_z**2]))

        # This is the exclusion zone radius, within which calculations of
        # intercept_normal_to() are automatically masked due to the ill-defined
        # geometry.
        self.exclusion = float(exclusion)
        self.r_exclusion = self.req * self.exclusion

        self.unmasked = self

        # Unique key for intercept calculations
        self.intercept_key = ('ellipsoid', self.origin.waypoint,
                                           self.frame.wayframe,
                                           tuple(self.radii),
                                           self.exclusion)

    def __getstate__(self):
        return (Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame),
                tuple(self.radii), self.exclusion)

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
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            axes        2 or 3, indicating whether to return the first two
                        coordinates (lon, lat) or all three (lon, lat, z) as
                        Scalars.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       optionally, the value of the coefficient p such that
                            ground + p * normal(ground) = pos;
                        ignored if the value is None (the default) or True.
            groundtrack True to return the intercept on the surface along with
                        the coordinates.

        Return:         a tuple of two to four items:
            lon         longitude at the surface in radians.
            lat         latitude at the surface in radians.
            z           vertical altitude in km normal to the surface; included
                        if axes == 3.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Use the quick solution for the body points if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            (track, p) = self.intercept_normal_to(pos, derivs=derivs, guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self.unsquash_sq
            track = pos.element_div(denom)

        # Derive the coordinates
        track_unsquashed = track.element_mul(self.unsquash)
        (x,y,z) = track_unsquashed.to_scalars()
        lat = (z/self.req).arcsin()
        lon = y.arctan2(x) % Scalar.TWOPI

        results = (lon, lat)

        if axes == 3:
            r = (pos - track).norm() * p.sign()
            results += (r,)

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
                lon     longitude at the surface in radians.
                lat     latitude at the surface in radians.
                z       vertical altitude in km normal to the body surface.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.
            groundtrack True to include the associated groundtrack points on the
                        body surface in the returned result.

        Return:         pos or (pos, track), where
            pos         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        # Determine groundtrack
        lon = Scalar.as_scalar(coords[0], derivs)
        lat = Scalar.as_scalar(coords[1], derivs)
        track_unsquashed = Vector3.from_ra_dec_length(lon, lat, self.req)
        track = track_unsquashed.element_mul(self.squash)

        # Assemble results
        if len(coords) == 2:
            results = (track, track)

        else:
            # Add the z-component
            normal = self.normal(track)
            results = (track + (coords[2] / normal.norm()) * normal, track)

        if groundtrack:
            return results

        return results[0]

    #===========================================================================
    def position_is_inside(self, pos, obs=None, time=None):
        """Where positions are inside the surface.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            obs         observer position as a Vector3 relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.

        Return:         Boolean True where positions are inside the surface
        """

        unsquashed = Vector3.as_vector3(pos).element_mul(self.unsquash)
        return unsquashed.norm() < self.radii[0]

    #===========================================================================
    def intercept(self, obs, los, time=None, direction='dep', derivs=False,
                                  guess=None, hints=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3 relative to this
                        surface's origin and frame.
            los         line of sight as a Vector3 in this surface's frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface.
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
            hints       the input value of hints, included if it is not None.
        """

        # Convert to Vector3 and un-squash
        obs = Vector3.as_vector3(obs, recursive=derivs)
        los = Vector3.as_vector3(los, recursive=derivs)

        obs_unsquashed = obs.element_mul(self.unsquash)
        los_unsquashed = los.element_mul(self.unsquash)

        # Solve for the intercept distance, masking lines of sight that miss
        #   pos = obs + t * los
        #   pos**2 = radius**2 [after "unsquash"]
        #
        # dot(obs,obs) + 2 * t * dot(obs,los) + t**2 * dot(los,los) = radius**2
        #
        # Use the quadratic formula to solve for t...
        #
        # a = los_unsquashed.dot(los_unsquashed)
        # b = los_unsquashed.dot(obs_unsquashed) * 2.
        # c = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        # d = b**2 - 4. * a * c
        #
        # Case 1: For photons departing from the surface and arriving at the
        # observer, we expect b > 0 (because dot(los,obs) must be positive for a
        # solution to exist) and we expect t < 0 (for an earlier time). In this
        # case, we seek the greater value of t, which corresponds to the surface
        # point closest to the observer.
        #
        # Case 2: For photons arriving at the surface, we expect b < 0 and
        # t > 0. In this case, we seek the lesser value of t, corresponding to
        # the point on the surface facing the source.
        #
        # However, also note that we need this method to work correctly even for
        # observers located "inside" the surface (where c < 0). This case is not
        # physical, but it can occur during iterations of _solve_photon_by_los.
        #
        # Case 1: If c < 0, we still seek the lesser value of t, but it will be
        # positive. In summary:
        #   t = (-b + sqrt(d)) / (2*a)
        # (because a is always positive) or, equivalently
        #   t = (-2*c) / (b + sqrt(d))
        # Of these two options, the second is preferred because, when outside
        # the body, it avoids the partial cancellation of -b and sqrt(d).
        #
        # Case 2: If c < 0, we still seek the greater value of t, but it will
        # be negative. In summary:
        #   t = (-b + sqrt(d)) / (2*a)
        # This is the preferred solution, because b and sqrt(d) usually have
        # opposite signs, so they generally do not cancel.

        # This is the same formula as above, but avoids a few multiplies by 2
        a      = los_unsquashed.dot(los_unsquashed)
        b_div2 = los_unsquashed.dot(obs_unsquashed)
        c      = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        d_div4 = b_div2**2 - a * c

        if direction == 'dep':                  # Case 1
            t = -c / (b_div2 + d_div4.sqrt())
        else:                                   # Case 2
            t = (d_div4.sqrt() - b_div2) / a

        pos = obs + t*los

        if hints is not None:
            return (pos, t, hints)

        return (pos, t)

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

        pos = Vector3.as_vector3(pos, derivs)
        return pos.element_mul(self.unsquash_sq)

    #===========================================================================
    def intercept_with_normal(self, normal, time=None, derivs=False):
        """Surface point where the normal vector parallels the given vector.

        Input:
            normal      a Vector3 of normal vectors in this surface's frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            derivs      True to propagate derivatives in the normal vector into
                        the returned intercepts.

        Return:         a Vector3 of surface intercept points, in km. Where no
                        solution exists, the returned Vector3 will be masked.
        """

        normal = Vector3.as_vector3(normal, derivs)
        return normal.element_mul(self.squash).unit().element_mul(self.radii)

    #===========================================================================
    def intercept_normal_to(self, pos, time=None, direction='dep', derivs=False,
                                       guess=None):
        """Surface point whose normal vector passes through a given position.

        This function can have multiple values, in which case the nearest of the
        surface points should be the one returned.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface; ignored here.
            derivs      True to propagate derivatives in pos into the returned
                        intercepts.
            guess       optional initial guess at coefficient p such that:
                            intercept + p * normal(intercept) = pos
                        Use guess=True for the converged value of p to be
                        returned even if an initial guess is unavailable.

        Return:         intercept or (intercept, p), where
            intercept   a Vector3 of surface intercept points relative to this
                        surface's origin and frame, in km. Where no intercept
                        exists, the returned vector will be masked.
            p           the converged solution such that
                            intercept = pos + p * normal(intercept);
                        included if the input guess is not None.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        pos = self._apply_exclusion(pos)

        # We need to solve for p such that:
        #   cept + p * normal(cept) = pos
        # where
        #   normal(cept) = cept.element_mul(self.unsquash_sq)
        #
        # This is subject to the constraint that cept is the intercept point on
        # the surface, where
        #   cept_unsquashed = cept.element_mul(self.unsquash)
        # and
        #   cept_unsquashed.dot(cept_unsquashed) = self.req_sq
        #
        # Let:

        B = self.unsquash_y_sq
        C = self.unsquash_z_sq
        R = self.req_sq

        # Four equations with four unknowns:
        # cept_x + p * cept_x = pos_x
        # cept_y + p * cept_y * B = pos_y
        # cept_z + p * cept_z * C = pos_z
        # cept_x**2 + cept_y**2 * B + cept_z**2 * C = R
        #
        # Let:

        (pos_x, pos_y, pos_z) = pos.to_scalars()
        X = pos_x**2
        Y = pos_y**2 * B
        Z = pos_z**2 * C

        # Plug the first three into the fourth and rearrange:
        #
        # f(p) = (  X * ((1 + B*p) * (1 + C*p))**2
        #         + Y * ((1 + p) * (1 + C*p))**2
        #         + Z * ((1 + p) * (1 + B*p))**2
        #         - R * ((1 + p) * (1 + B*p) * (1 + C*p))**2)
        #
        # This is a sixth-order polynomial, which we need to solve for f(p) = 0.
        #
        # Using SymPy, this expands to:
        #
        # f(p) = -B**2*C**2*R*p**6
        #      + p**5*(-2*B**2*C**2*R - 2*B**2*C*R - 2*B*C**2*R)
        #      + p**4*(-B**2*C**2*R + B**2*C**2*X - 4*B**2*C*R - B**2*R
        #              + B**2*Z - 4*B*C**2*R - 4*B*C*R - C**2*R + C**2*Y)
        #      + p**3*(-2*B**2*C*R + 2*B**2*C*X - 2*B**2*R + 2*B**2*Z
        #              - 2*B*C**2*R + 2*B*C**2*X - 8*B*C*R - 2*B*R + 2*B*Z
        #              - 2*C**2*R + 2*C**2*Y - 2*C*R + 2*C*Y)
        #      + p**2*(-B**2*R + B**2*X + B**2*Z - 4*B*C*R + 4*B*C*X - 4*B*R
        #              + 4*B*Z - C**2*R + C**2*X + C**2*Y - 4*C*R + 4*C*Y - R
        #              + Y + Z)
        #      + p*(-2*B*R + 2*B*X + 2*B*Z - 2*C*R + 2*C*X + 2*C*Y - 2*R + 2*Y
        #           + 2*Z)
        #      - R + X + Y + Z
        #
        # Let f(p) = (((((f6*p + f5)*p + f4)*p + f3)*p + f2)*p + f1)*p + f0

        B2 = B**2
        C2 = C**2

        # For efficiency, we segregate all the array ops (involving X, Y, Z)
        f6 = -B2 * C2 * R
        f5 = -2 * R * (B2*C2 + B2*C + B*C2)
        f4 = (X * (B2*C2) + Y * C2 + Z * B2
              - R * (B2*C2 + 4*B2*C + 4*B*C2 + 4*B*C + B2 + C2))
        f3 = (X * (2*(B2*C + B*C2)) + Y * (2*(C2 + C)) + Z * (2*(B2 + B))
              - 2 * R * (B2*C + B*C2 + 4*B*C + B2 + C2 + B + C))
        f2 = (X * (B2 + 4*B*C + C2) + Y * (C2 + 4*C + 1) + Z * (B2 + 4*B + 1)
              - R * (B2 + 4*B*C + C2 + 4*B + 4*C + 1))
        f1 = (X * (2*B + 2*C) + Y * (2*C + 2) + Z * (2*B + 2)
              - 2 * R * (B + C + 1))
        f0 = X + Y + Z - R

        g5 = 6 * f6
        g4 = 5 * f5
        g3 = 4 * f4
        g2 = 3 * f3
        g1 = 2 * f2
        g0 = f1

        # Make an initial guess at p
        if isinstance(guess, (type(None), bool, np.bool_)):

            # Unsquash into coordinates where the surface is a sphere
            pos_unsq = pos.wod.element_mul(self.unsquash)   # without derivs!

            # Estimate the intercept point as on a straight line to the origin
            # (Note that this estimate is exact for points at the surface.)
            cept_guess_unsq = pos_unsq.with_norm(self.req)

            # Make a guess at the normal vector in unsquashed coordinates
            normal_guess_unsq = cept_guess_unsq.element_mul(self.unsquash_sq)

            # Estimate p for [cept + p * normal(cept) = pos] using norms
            p = ((pos_unsq.norm() - cept_guess_unsq.norm())
                 / normal_guess_unsq.norm())

        else:
            p = guess.wod.copy()

        # The precision of p should match the default geometric accuracy defined
        # by SURFACE_PHOTONS.km_precision. Set our precision goal on p
        # accordingly.
        km_scale = self.req
        precision = SURFACE_PHOTONS.km_precision / km_scale

        # Iterate until convergence stops
        max_dp = 1.e99
        converged = False

        # We typically need a few extra iterations to reach desired precision
        for count in range(SURFACE_PHOTONS.max_iterations + 10):

            # Calculate f and df/dp
            f = (((((f6*p + f5)*p + f4)*p + f3)*p + f2)*p + f1)*p + f0
            df_dp = ((((g5*p + g4)*p + g3)*p + g2)*p + g1)*p + g0

            # One step of Newton's method
            dp = f / df_dp
            p -= dp

            prev_max_dp = max_dp
            max_dp = dp.abs().max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Ellipsoid.DEBUG:
                LOGGING.convergence(
                            '%s.intercept_normal_to(): iter=%d; change[km]=%.6g'
                            % (type(self).__name__, count+1, max_dp * km_scale))

            if max_dp <= precision:
                converged = True
                break

            if max_dp >= prev_max_dp:
                break

        if not converged:
            LOGGING.warn('%s.intercept_normal_to() did not converge: '
                         'iter=%d; change[km]=%.6g'
                         % (type(self).__name__, count+1, max_dp * km_scale))

        cept_x = pos_x / (1 + p)
        cept_y = pos_y / (1 + B * p)
        cept_z = pos_z / (1 + C * p)
        cept = Vector3.from_scalars(cept_x, cept_y, cept_z)

        if guess is None:
            return cept
        else:
            return (cept, p)

    #===========================================================================
    def _apply_exclusion(self, pos):
        """This internal method is used by intercept_normal_to() to exclude any
        positions that fall too close to the center of the surface. The math
        is poorly-behaved in this region.

        (1) It sets the mask on any of these points to True.
        (2) It sets the magnitude of any of these points to the edge of the
            exclusion zone, in order to avoid runtime errors in the math
            libraries.
        """

        pos_unsquashed = pos.element_mul(self.unsquash)
        norm_sq = pos_unsquashed.wod.norm_sq()
        mask = (norm_sq < self.r_exclusion**2)
        if not mask.any():
            return pos

        rescale = Scalar.maximum(1., self.r_exclusion / norm_sq.sqrt())
        return (pos * rescale).remask_or(mask)

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.squash_y).arctan2(lon.cos())

    #===========================================================================
    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.unsquash_y).arctan2(lon.cos())

    #===========================================================================
    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.unsquash_y).arctan2(lon.cos())

    #===========================================================================
    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.squash_y).arctan2(lon.cos())

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        denom = (lon.cos()**2 + (lon.sin() * self.squash_y)**2).sqrt()

        return (lat.tan() * self.squash_z / denom).arctan()

    #===========================================================================
    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to internal ellipsoid latitude.

        Input:
            lat         planetocentric latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          squashed latitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        factor = (lon.cos()**2 + (lon.sin() * self.squash_y)**2).sqrt()

        return (lat.tan() * self.unsquash_z * factor).arctan()

    #===========================================================================
    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetographic.

        Input:
            lat         squashed latitide, radians.
            lon         squashed longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        denom = (lon.cos()**2 + (lon.sin() * self.unsquash_y)**2).sqrt()

        return (lat.tan() * self.unsquash_z / denom).arctan()

    #===========================================================================
    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert a planetographic latitude to internal ellipsoid latitude.

        Input:
            lat         planetographic latitide, radians.
            lon         planetographic longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          squashed latitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        factor = (lon.cos()**2 + (lon.sin() * self.unsquash_y)**2).sqrt()

        return (lat.tan() * self.squash_z * factor).arctan()

################################################################################
