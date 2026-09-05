##########################################################################################
# oops/surface/ellipsoid.py
##########################################################################################

import numpy as np

from polymath              import Boolean, Matrix, Scalar, Vector3
from oops.config           import SURFACE_PHOTONS, LOGGING
from oops.constants        import HALFPI, TWOPI
from oops.frame.frame_     import Frame
from oops.path.path_       import Path
from oops.surface.surface_ import Surface

# A `z`-coordinate measured normal to the surface becomes poorly behaved near the evolute
# of the ellipsoid. For radii `a >= b >= c`, the smallest radius of curvature anywhere on
# the surface is `c^2/a`, at the ends of the longest axis, so the evolute reaches to
# within that distance of the surface and `z <= -c^2/a` is ruled out. We exclude `z < -g
# c^2/a`, keeping a margin below that depth. In "unsquashed" coordinates, where the
# surface becomes a sphere of radius `a`, this excludes a sphere about the center of
# radius `a + z = a - g c^2/a`.
#
# A `g` factor of 1 would put the boundary on the evolute itself, where
# `intercept_normal_to()` fails outright. Accuracy degrades well before that: measured
# over a range of axis ratios, the intercepts returned for unmasked interior positions
# hold to a few parts in 1e14 of the radius up to a factor near 0.55, and lose several
# digits by 0.8. This value keeps the full precision with a wide margin below the evolute.
#
# For a body flatter than about `c/a = 0.83`, the evolute also reaches beyond this sphere
# along the polar axis, but only outside the surface, where masking would discard
# legitimate positions. Those positions are limited by the convergence of
# `intercept_normal_to()` rather than by this zone, so no exclusion radius can address
# them.
_EXCLUSION_FACTOR = 0.5     # This is `g` in the above discussion


class Ellipsoid(Surface):
    """An ellipsoidal surface centered on a path and fixed within a frame.

    The short radius of the ellipsoid is oriented along the *z*-axis of the frame and the
    long radius is along the *x*-axis.

    The coordinates defining the surface grid are `(longitude, latitude)`. Both are based
    on the assumption that a spherical body has been "squashed" along the Y- and Z-axes.
    The latitudes and longitudes defined in this manner are neither planetocentric nor
    planetographic; functions are provided to perform conversions to either choice.
    Longitudes are measured in a right-handed manner, increasing toward the east; values
    range from 0 to 2*pi.

    The third coordinate is `z`, which measures vertical distance in km along the normal
    vector from the surface.
    """

    COORDINATE_TYPE = 'spherical'
    COORDINATE_NAMES = ('longitude', 'latitude', 'elevation')
    COORDINATE_ABBREVS = ('lon', 'lat', 'z')
    COORDINATE_RANGES = ((0, TWOPI), (-HALFPI, HALFPI), (None, None))
    IS_VIRTUAL = False
    HAS_INTERIOR = True

    _DEBUG = False       # True for convergence testing in intercept_normal_to()

    def __init__(self, origin, frame, radii):
        """Constructor for an Ellipsoid object.

        Parameters:
            origin (Path or str): The Path or the ID of the Path defining the center of
                the ellipsoid.
            frame (Frame or str): The Frame or the ID of the Frame in which the
                ellipsoid is fixed, with the shortest radius of the ellipsoid along the
                *z*-axis and the longest radius along the *x*-axis.
            radii (tuple[float, float, float]): `(a, b, c)`, the radii from longest to
                shortest, in km.
        """

        self.origin = Path.as_waypoint(origin)
        self.frame  = Frame.as_wayframe(frame)

        self._radii    = np.asarray(radii, dtype=np.float64)
        self._radii_sq = self._radii**2
        self._req      = self._radii[0]
        self._req_sq   = self._req**2
        self._rpol     = self._radii[2]

        self._squash_y       = self._radii[1] / self._radii[0]
        self._squash_y_sq    = self._squash_y**2
        self._unsquash_y     = self._radii[0] / self._radii[1]
        self._unsquash_y_sq  = self._unsquash_y**2

        self._squash_z       = self._radii[2] / self._radii[0]
        self._squash_z_sq    = self._squash_z**2
        self._unsquash_z     = self._radii[0] / self._radii[2]
        self._unsquash_z_sq  = self._unsquash_z**2

        self._squash         = Vector3((1., self._squash_y, self._squash_z))
        self._squash_sq      = self._squash.element_mul(self._squash)
        self._unsquash       = Vector3((1., 1./self._squash_y, 1./self._squash_z))
        self._unsquash_sq    = self._unsquash.element_mul(self._unsquash)

        self._unsquash_sq_2d = Matrix(([1.,0.,0.],
                                       [0.,self._unsquash_y**2,0.],
                                       [0.,0.,self._unsquash_z**2]))

        # This is the exclusion zone radius, within which calculations of
        # intercept_normal_to() are automatically masked due to the ill-defined geometry.
        self._r_exclusion = self._req * (1. - _EXCLUSION_FACTOR * self._squash_z_sq)

        self.unmasked = self

        # Unique key for intercept calculations
        self.intercept_key = ('ellipsoid', self.origin.waypoint, self.frame.wayframe,
                                           tuple(self._radii))

    @property
    def radii(self):
        """The three radii of the ellipsoid in km, from longest to shortest.

        Returns:
            numpy.ndarray: The (a, b, c) radii, in km.
        """

        return self._radii

    @property
    def unsquash_sq(self):
        """The squared scale factors that unsquash a vector into spherical coordinates.

        Multiplying a vector by these factors element by element converts it from the
        ellipsoid's frame to the frame in which the body is a sphere.

        Returns:
            Vector3: The squared unsquash factors.
        """

        return self._unsquash_sq

    def __getstate__(self):
        self.refresh()
        return (Path.as_primary_path(self.origin), Frame.as_primary_frame(self.frame),
                tuple(self._radii))

    def __setstate__(self, state):
        (origin, frame, radii) = state
        self.__init__(origin, frame, radii)
        self.freeze()

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (lon, lat) or all three (lon, lat, z) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Scalar, optional): Optionally, the value of the coefficient `p` such
                that ``ground + p * normal(ground) = pos``. If it is not None, the
                converged value of `p` is appended to the returned tuple; use `hints=True`
                if you lack an initial value but require the new value to be returned.
            groundtrack (bool, optional): True to append the intercept point on the
                surface to the returned tuple.

        Returns:
            tuple[Scalar, ...]: Two to five items:

            * `lon` (Scalar): Longitude at the surface in radians.
            * `lat` (Scalar): Latitude at the surface in radians.
            * `z` (Scalar): Vertical altitude in km normal to the surface; included if
              `axes` == 3.
            * `p` (Scalar): The converged coefficient; included if the input value of
              `hints` is not None.
            * `track` (Vector3): Intercept point on the surface (where *z == 0*); included
              if `groundtrack` is True.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Use the quick solution for the body points if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            (track, p) = self.intercept_normal_to(pos, derivs=derivs, guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self._unsquash_sq
            track = pos.element_div(denom)

        # Derive the coordinates
        track_unsquashed = track.element_mul(self._unsquash)
        (x,y,z) = track_unsquashed.to_scalars()
        lat = (z/self._req).arcsin()
        lon = y.arctan2(x) % Scalar.TWOPI

        results = (lon, lat)

        if axes == 3:
            r = (pos - track).norm() * p.sign()
            results += (r,)

        if hints is not None:
            results += (p,)

        if groundtrack:
            results += (track,)

        return results

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None, groundtrack=False):
        """The position at the given surface coordinates.

        Parameters:
            coords (tuple[Scalar, ...]): Two or three Scalars defining coordinates at
                or near this surface. These can have different shapes, but must be
                broadcastable to a common shape.

                * `lon` (rad): Longitude at the surface.
                * `lat` (rad): Latitude at the surface.
                * `z` (km, optional): Vertical altitude normal to the body surface.

            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives inside the
                coordinates and obs into the returned position vectors.
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
            * `track` (Vector3): Intercept point on the surface (where *z == 0*); included
              if `groundtrack` is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        # Determine groundtrack
        lon = Scalar.as_scalar(coords[0], recursive=derivs)
        lat = Scalar.as_scalar(coords[1], recursive=derivs)
        track_unsquashed = Vector3.from_ra_dec_length(lon, lat, self._req)
        track = track_unsquashed.element_mul(self._squash)

        # Assemble results
        if len(coords) == 2:
            results = (track, track)

        else:
            # Add the z-component. The normal direction varies with lon and lat, so its
            # derivatives belong in the result along with those of the groundtrack.
            normal = self.normal(track, derivs=derivs)
            results = (track + (coords[2] / normal.norm()) * normal, track)

        extras = ()
        if hints is not None:
            extras += (hints,)

        if groundtrack:
            extras += (results[1],)

        if extras:
            return (results[0],) + extras

        return results[0]

    def position_is_inside(self, pos, *, obs=None, time=None):
        """Where positions are inside the surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's
                origin and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.

        Returns:
            Boolean: True where positions are inside the Surface.
        """

        unsquashed = Vector3.as_vector3(pos).element_mul(self._unsquash)
        return unsquashed.norm() < self._radii[0]

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False,
                  guess=None, hints=None):
        """The position where a specified line of sight intercepts the Surface.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this Surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this Surface's frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface.
            derivs (bool, optional): True to propagate any derivatives inside obs and los
                into the returned intercept point.
            guess (Scalar, optional): Unused.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            tuple[Vector3, Scalar[, Any]]: `(pos, t)` or `(pos, t, hints)`, where:

            * `pos` (Vector3): Intercept points on the Surface relative to this surface's
              origin and frame, in km.
            * `t` (Scalar): Such that ``intercept = obs + t * los``.
            * `hints` (Any): The input value of `hints`, included if it is not None.

        Raises:
            ValueError: If `direction` is neither "arr" nor "dep".
        """

        # Convert to Vector3 and un-squash
        obs = Vector3.as_vector3(obs, recursive=derivs)
        los = Vector3.as_vector3(los, recursive=derivs)

        obs_unsquashed = obs.element_mul(self._unsquash)
        los_unsquashed = los.element_mul(self._unsquash)

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
        # c = obs_unsquashed.dot(obs_unsquashed) - self._req_sq
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
        c      = obs_unsquashed.dot(obs_unsquashed) - self._req_sq
        d_div4 = b_div2**2 - a * c

        if direction == 'dep':                  # Case 1
            t = -c / (b_div2 + d_div4.sqrt())
        elif direction == 'arr':                # Case 2
            t = (d_div4.sqrt() - b_div2) / a
        else:
            raise ValueError('invalid direction: ' + repr(direction))

        pos = obs + t*los

        if hints is not None:
            return (pos, t, hints)

        return (pos, t)

    def normal(self, pos, *, obs=None, time=None, derivs=False, hints=None):
        """The normal vector at a position at or near a surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives of `pos` into the
                returned normal vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Directions normal to the Surface that pass
            through the position, optionally followed by `hints`. Vector lengths are
            arbitrary, and the input value of `hints` is returned if it is not None.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        perp = pos.element_mul(self._unsquash_sq)

        if hints is not None:
            return (perp, hints)

        return perp

    def intercept_with_normal(self, normal, *, obs=None, time=None, derivs=False,
                              hints=None):
        """Surface point where the normal vector parallels the given vector.

        Parameters:
            normal (Vector3): Normal vectors in this Surface's frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate derivatives in the normal vector
                into the returned intercepts.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Surface intercept points in km, optionally
            followed by `hints`. Where no solution exists, values are masked, and the
            input value of `hints` is returned if it is not None.
        """

        normal = Vector3.as_vector3(normal, recursive=derivs)
        cept = normal.element_mul(self._squash).unit().element_mul(self._radii)

        if hints is not None:
            return (cept, hints)

        return cept

    def intercept_normal_to(self, pos, *, obs=None, time=None, direction='dep',
                            derivs=False, guess=None, hints=None):
        """Surface point whose normal vector passes through a given position.

        This function can have multiple values, in which case the nearest of the surface
        points should be the one returned.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored here.
            derivs (bool, optional): True to propagate derivatives in pos into the
                returned intercepts.
            guess (Scalar, optional): Optional initial guess at coefficient `p` such that
                ``intercept + p * normal(intercept) = pos``. Use `guess=True` for the
                converged value of `p` to be returned even if an initial guess is
                unavailable.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple: `intercept` or `(intercept[, p][, hints])`, where:

            * `intercept` (Vector3): Surface intercept points relative to this surface's
              origin and frame, in km. Where no intercept exists, values are masked.
            * `p` (Scalar): The converged solution such that
              ``intercept + p * normal(intercept) = pos``; included if the input value
              of `guess` is not None.
            * `hints` (Any): The input value of `hints`, included if it is not None.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        pos = self._apply_exclusion(pos)

        # We need to solve for p such that:
        #   cept + p * normal(cept) = pos
        # where
        #   normal(cept) = cept.element_mul(self._unsquash_sq)
        #
        # This is subject to the constraint that cept is the intercept point on
        # the surface, where
        #   cept_unsquashed = cept.element_mul(self._unsquash)
        # and
        #   cept_unsquashed.dot(cept_unsquashed) = self._req_sq
        #
        # Let:

        B = self._unsquash_y_sq
        C = self._unsquash_z_sq
        R = self._req_sq

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
            pos_unsq = pos.wod.element_mul(self._unsquash)   # without derivs!

            # Estimate the intercept point as on a straight line to the origin
            # (Note that this estimate is exact for points at the surface.)
            cept_guess_unsq = pos_unsq.with_norm(self._req)

            # Make a guess at the normal vector in unsquashed coordinates
            normal_guess_unsq = cept_guess_unsq.element_mul(self._unsquash_sq)

            # Estimate p for [cept + p * normal(cept) = pos] using norms
            p = ((pos_unsq.norm() - cept_guess_unsq.norm())
                 / normal_guess_unsq.norm())

        else:
            p = guess.wod.copy()

        # The precision of p should match the default geometric accuracy defined
        # by SURFACE_PHOTONS.km_precision. Set our precision goal on p
        # accordingly.
        km_scale = self._req
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

            if LOGGING.surface_iterations or Ellipsoid._DEBUG:
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

        results = (cept,)
        if guess is not None:
            results += (p,)

        if hints is not None:
            results += (hints,)

        if len(results) == 1:
            return cept

        return results

    def _apply_exclusion(self, pos):
        """The given positions, with those too close to the center excluded.

        Used by `intercept_normal_to`, where the math is poorly behaved close to the
        center. Positions inside the exclusion zone are masked, and their magnitudes are
        set to the edge of that zone so that the math libraries do not raise runtime
        errors.

        Parameters:
            pos (Vector3): Positions relative to this Surface's origin and frame.

        Returns:
            Vector3: The given positions, masked and rescaled where they fall inside the
            exclusion zone.
        """

        pos_unsquashed = pos.element_mul(self._unsquash)
        norm_sq = pos_unsquashed.wod.norm_sq()
        mask = Boolean.as_boolean(norm_sq < self._r_exclusion**2)
        if not mask.any():
            return pos

        rescale = Scalar.maximum(1., self._r_exclusion / norm_sq.sqrt())
        return (pos * rescale).remask_or(mask)

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self._squash_y).arctan2(lon.cos())

    def lon_from_centric(self, lon, *, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self._unsquash_y).arctan2(lon.cos())

    def lon_to_graphic(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetographic longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self._unsquash_y).arctan2(lon.cos())

    def lon_from_graphic(self, lon, *, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self._squash_y).arctan2(lon.cos())

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        denom = (lon.cos()**2 + (lon.sin() * self._squash_y)**2).sqrt()

        return (lat.tan() * self._squash_z / denom).arctan()

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        factor = (lon.cos()**2 + (lon.sin() * self._squash_y)**2).sqrt()

        return (lat.tan() * self._unsquash_z * factor).arctan()

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        denom = (lon.cos()**2 + (lon.sin() * self._unsquash_y)**2).sqrt()

        return (lat.tan() * self._unsquash_z / denom).arctan()

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

        lon = Scalar.as_scalar(lon, recursive=derivs)
        lat = Scalar.as_scalar(lat, recursive=derivs)

        factor = (lon.cos()**2 + (lon.sin() * self._unsquash_y)**2).sqrt()

        return (lat.tan() * self._squash_z * factor).arctan()

##########################################################################################
