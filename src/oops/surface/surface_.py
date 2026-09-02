##########################################################################################
# oops/surface/surface_.py: Abstract class Surface
##########################################################################################

import numbers
import numpy as np

from polymath     import Boolean, Scalar, Vector3
from oops.event   import Event
from oops.mutable import Mutable


class Surface(Mutable):
    """An abstract class describing a 2-D object that moves and rotates in space.

    A surface employs an internal coordinate system, not necessarily rectangular, in which
    two primary coordinates define locations on the surface, and an optional third
    coordinate can define points above or below that surface.

    Attributes:
        origin (Path): The waypoint of the path defining the surface's center.
        frame (Frame): The wayframe of the frame in which the surface is defined.
        unmasked (Surface): An unmasked version of this surface. If the surface has no
            mask, this returns self.
        intercept_key (tuple): A unique, immutable key that defines the surface. Some
            surface classes are identical except for a mask or coordinate definition;
            those classes return the same intercept key.
        IS_VIRTUAL (bool): True if this surface is virtual. A virtual surface is one that
            is only defined from the viewpoint of an observer, such as a Limb or Ansa
            point. It has no physical presence.
        IS_TIME_DEPENDENT (bool): True if the 3-D shape of the surface varies with time.
            Use False if a surface is rigid. Note that time dependence in the Path and/or
            Frame doesn't count.
        HAS_INTERIOR (bool): True if this surface has an inside and an outside. A Spheroid
            has an interior; a Ring does not.
        COORDINATE_TYPE (str): One of "rectangular", "cylindrical", "spherical", "polar",
            or "limb".
        COORDINATE_NAMES (tuple[str, str, str]): Names of the three coordinates.
        COORDINATE_ABBREVS (tuple[str, str, str]): Short abbreviations for the three
            coordinates.
        COORDINATE_RANGES (tuple[tuple[float or None, float or None], ...]): Numeric
            ranges for the three components: `(None, None)` for no limits, `(0, None)` for
            non-negative, `(0, 2*pi)` for a cyclic angle, or any other pair of floats to
            define the specific limits of the range.
    """

    # Default attributes; override as needed
    IS_VIRTUAL = False
    IS_TIME_DEPENDENT = False
    HAS_INTERIOR = False
    COORDINATE_TYPE = 'rectangular'
    COORDINATE_NAMES = ('x', 'y', 'z')
    COORDINATE_ABBREVS = ('x', 'y', 'z')
    COORDINATE_RANGES = ((None, None), (None, None), (None, None))

    # Class constant to avoid circular references
    _Body = None            # filled in by body.py

    # Class constants to override where derivs are undefined
    _coords_from_vector3_DERIVS_ARE_IMPLEMENTED = True
    _vector3_from_coords_DERIVS_ARE_IMPLEMENTED = True
    _intercept_DERIVS_ARE_IMPLEMENTED = True
    _normal_DERIVS_ARE_IMPLEMENTED = True
    _intercept_with_normal_DERIVS_ARE_IMPLEMENTED = True
    _intercept_normal_to_DERIVS_ARE_IMPLEMENTED = True

    def _coords_from_vector3_check(self, axes):
        """Validate axes as equal to 2 or 3."""

        if not isinstance(axes, numbers.Integral) or axes not in (2, 3):
            raise ValueError(f'axes must be 2 or 3 in {type(self).__name__}'
                             '.coords_from_vector3()')

    def _vector3_from_coords_check(self, coords):
        """Validate coords as a tuple of 2 or 3 Scalars."""

        if not isinstance(coords, (tuple, list)) or len(coords) not in (2, 3):
            raise ValueError(f'2 or 3 coords required in {type(self).__name__}'
                             '.vector3_from_coords()')

    ######################################################################################
    # Each subclass must override...
    ######################################################################################

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to its origin and
                frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                the Surface is time-variable.
            axes (int, optional): 2 or 3, indicating whether to return a tuple of two or
                three Scalar objects.
            derivs (bool, optional): True to propagate any derivatives inside `pos` and
                `obs` into the returned coordinates.
            hints (Any, optional): Optional data that might be useful to carry over from
                one call to the next. If not None, `hints` values are appended to the
                returned tuple. Use `hints=True` if you lack an initial value but require
                the new value to be returned.

        Returns:
            tuple: Two or three Scalar coordinate values depending on the input value of
            `axes`, optionally followed by `hints`.
        """

        raise NotImplementedError(f'{type(self).__name__}.coords_from_vector3 is not '
                                  'implemented')

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None):
        """The position where a point with the given coordinates falls relative to this
        Surface's origin and frame.

        Parameters:
            coords (tuple[Scalar, ...]): Two or three Scalars defining coordinates at or
                near this Surface. These can have different shapes, but must be
                broadcastable to a common shape.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at which to evaluate the surface; ignored unless
                the surface is time-variable.
            derivs (bool, optional): True to propagate any derivatives inside `coords`
                and `obs` into the returned position vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next. If not None, `hints` values are appended to the returned
                tuple. Use `hints=True` if you lack an initial value but require the new
                value to be returned.

        Returns:
            Vector3 or tuple[Vector3, Any]: Points defined by the coordinates, relative to
            this Surface's origin and frame, optionally followed by `hints`. For Surface
            subclasses that do not use hints, the input value of `hints` is returned.
        """

        raise NotImplementedError(f'{type(self).__name__}.vector3_from_coords() is not '
                                  'implemented')

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False, guess=None,
                  hints=None):
        """The position where a specified line of sight intercepts the Surface.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this Surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this Surface's frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                the Surface is time-variable.
            direction (str, optional): "arr" for a photon arriving at the Surface; "dep"
                for a photon departing from the Surface. Needed for closed surfaces that
                have two intercept points, one inward-facing and one outward-facing;
                ignored otherwise.
            derivs (bool, optional): True to propagate any derivatives inside `obs` and
                `los` into the returned intercept point.
            guess (Scalar, optional): Optional initial guess at the coefficient `t` such
                that `intercept = obs + t * los`.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next. If not None, `hints` values are appended to the returned
                tuple. Use `hints=True` if you lack an initial value but require the new
                value to be returned.

        Returns:
            tuple[Vector3, Scalar[, Any]]: `(pos, t)` or `(pos, t, hints)`, where:

            * `pos` (Vector3): Intercept points on the Surface relative to its origin
              and frame, in km.
            * `t` (Scalar): Value such that `intercept = obs + t * los`.
            * `hints` (Any): Latest version of the `hints` values to be fed into a
              subsequent call. If the subclass does not use hints, the input value of
              `hints` is included if it is not None.
        """

        raise NotImplementedError(f'{type(self).__name__}.intercept() is not implemented')

    def normal(self, pos, *, obs=None, time=None, derivs=False, hints=None):
        """The normal vector at a position at or near this Surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to its origin and
                frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                the Surface is time-variable.
            derivs (bool, optional): True to propagate any derivatives of `pos` into the
                returned normal vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next. If not None, `hints` values are appended to the returned
                tuple. Use `hints=True` if you lack an initial value but require the new
                value to be returned.

        Returns:
            Vector3 or tuple[Vector3, Any]: Directions normal to the Surface that pass
            through the position, optionally followed by `hints`. Vector lengths are
            arbitrary.
        """

        raise NotImplementedError(f'{type(self).__name__}.normal() is not implemented')

    ######################################################################################
    # Optional Methods...
    ######################################################################################

    def intercept_with_normal(self, normal, *, obs=None, time=None, derivs=False,
                              hints=None):
        """Surface point where the outward normal vector parallels the given vector.

        Parameters:
            normal (Vector3): Normal vector in the Surface's frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at the Surface; ignored unless the Surface is
                time-variable.
            derivs (bool, optional): True to propagate derivatives in the normal vector
                into the returned surface points.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next. If not None, `hints` values are appended to the returned
                tuple. Use `hints=True` if you lack an initial value but require the new
                value to be returned.

        Returns:
            Vector3 or tuple[Vector3, Any]: Surface intercept points in km, optionally
            followed by `hints`. Where no solution exists, values of the intercept point
            are masked.
        """

        raise NotImplementedError(f'{type(self).__name__}.intercept_with_normal() is not '
                                  'implemented')

    def intercept_normal_to(self, pos, *, obs=None, time=None, direction='dep',
                            derivs=False, guess=None, hints=None):
        """Surface point whose normal vector passes through a given position.

        This function can have multiple values, in which case the nearest of the surface
        points should be the one returned.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to its origin and
                frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at the Surface; ignored unless the Surface is
                time-variable.
            direction (str, optional): "arr" for a photon arriving at the Surface; "dep"
                for a photon departing from the Surface. Needed for closed surfaces that
                have two intercept points; ignored otherwise.
            derivs (bool, optional): True to propagate derivatives in `pos` and `obs` into
                the returned intercept points.
            guess (Scalar, optional): Optional initial guess at the coefficient `p` such
                that `intercept + p * normal(intercept) = pos`. If provided, the converged
                value of `p` is included in the returned results; use `guess=True` to
                include this in the return even if an initial guess is not available.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next. If not None, `hints` values are appended to the returned
                tuple. Use `hints=True` if you lack an initial value but require the new
                value to be returned.

        Returns:
            Vector3 or tuple[Vector3[, Scalar][, Any]]: `intercept` or tuple of up to
            three values, depending on the input values of `guess` and `hints`.

            * `intercept` (Vector3): The Surface intercept points, in km. Where no
              solution exists, values are masked.
            * `p` (Scalar): The converged solution where::

                 intercept + p * normal(intercept) = pos

              This is included if `guess` is not None. For subclasses that do not use a
              guess, the input value of `guess` is returned.
            * `hints` (Any): Latest version of any hint values; included if the input
              value of `hints` is not None. For Surface subclasses that do not use hints,
              the input value of `hints` is returned.
        """

        raise NotImplementedError(f'{type(self).__name__}.intercept_normal_to() is not '
                                  'implemented')

    def velocity(self, pos, *, obs=None, time=None):
        """The local velocity vector at a point within this Surface.

        This can be used to describe the orbital motion of ring particles or local wind
        speeds on a planet.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to its origin and
                frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                the Surface is time-variable.

        Returns:
            Vector3: Velocities, in units of km/s.
        """

        return Vector3.ZERO

    def position_is_inside(self, pos, *, obs=None, time=None):
        """True where positions are inside this Surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to its origin and
                frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                the Surface is time-variable.

        Returns:
            Boolean: True where positions are inside the Surface. For subclasses
            that have no interior, such as RingPlane, a single value of Boolean False is
            always returned.

        Raises:
            NotImplementedError: If the Surface subclass has an interior but this method
                is not implemented.
        """

        if self.HAS_INTERIOR:
            raise NotImplementedError(f'{type(self).__name__}.position_is_inside() is '
                                      'not implemented')

        return Boolean.FALSE

    ######################################################################################
    # Support for surfaces derived from other surfaces. E.g., surfaces using different
    # coordinates or with boundaries applied.
    ######################################################################################

    def reference(self):
        """The reference Surface for this one.

        Returns:
            Surface: The Surface from which this one is derived. The default is to return
            this Surface itself.
        """

        return self     # default is to return self

    ######################################################################################
    # Event-coordinate conversions. Generally should not require overrides
    ######################################################################################

    def coords_of_event(self, event, *, obs=None, axes=3, derivs=False):
        """Coordinate values associated with an event near the Surface.

        Parameters:
            event (Event): An event occurring at or near the Surface.
            obs (Event, optional): Observing event, which may occur at a different time.
            axes (int, optional): 2 or 3, indicating whether to return a tuple of two or
                three Scalar objects.
            derivs (bool, optional): If True, then all derivatives are carried forward
                into the event; if False, only time derivatives are included.

        Returns:
            tuple[Scalar, ...]: Two or three unitless Scalars, one for each coordinate.
        """

        # Locate the events WRT the surface frame
        cept_in_frame = event.wrt(self.origin, self.frame, derivs=derivs).state
        obs_in_frame = obs and obs.wrt(self.origin, self.frame, derivs=derivs).state

        # Evaluate the coords and optional derivatives
        # The `hints` attribute is filled in by the _photon_solver methods.
        hints = event.hints if hasattr(event, 'hints') else None
        result = self.coords_from_vector3(cept_in_frame, obs=obs_in_frame,
                                          time=event.time, axes=axes, derivs=True,
                                          hints=hints)
        return result[:axes]

    def apply_coords_to_event(self, event, *, obs=None, axes=3, derivs=True):
        """A shallow copy of the given Event with subfields for the coordinates.

        Parameters:
            event (Event): An event occurring at or near this Surface.
            obs (Event, optional): Observing event, which may occur at a different time.
            axes (int, optional): 2 or 3, indicating whether to add two or three Scalar
                subfields.
            derivs (bool, optional): If True, then all derivatives are carried forward
                into the event; if False, only time derivatives are included.

        Returns:
            Event: Clone of `event` with new attributes `coord1`, `coord2`, and
            optionally `coord3`.
        """

        coords = self.coords_of_event(event, obs=obs, axes=axes, derivs=derivs)

        event = event.copy(omit=('coord1', 'coord2', 'coord3'))
        event.insert_subfield('coord1', coords[0])
        event.insert_subfield('coord2', coords[1])
        if axes > 2:
            event.insert_subfield('coord3', coords[2])

        if np.any(coords[0].mask):
            event = event.mask_where(coords[0].mask)

        return event

    def event_at_coords(self, time, coords, *, obs=None, derivs=False):
        """An Event constructed from a time and coordinates in this Surface's internal
        coordinate system.

        Parameters:
            time (Scalar): Time in seconds TDB.
            coords (tuple[Scalar, ...]): 2 or 3 coordinates.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame. Ignored for solid Surfaces but needed for virtual Surfaces.
            derivs (bool, optional): If True, then all derivatives are carried forward
                into the event; if False, only time derivatives are included.

        Returns:
            Event: An event object relative to the origin and frame of this Surface.
        """

        # Interpret coords
        if len(coords) == 2:
            (coord1, coord2) = coords
            coord3 = Scalar.ZERO
        else:
            (coord1, coord2, coord3) = coords

        # Strip derivatives if necessary, but not d_dt
        if not derivs:
            coord1 = coord1.without_derivs(preserve='t')
            coord2 = coord2.without_derivs(preserve='t')
            coord3 = coord3.without_derivs(preserve='t')

            if obs is not None:
                obs = obs.without_derivs(preserve='t')

        # Determine position and velocity
        state = self.vector3_from_coords((coord1, coord2, coord3), obs=obs, time=time,
                                         derivs=True)

        # Return the event
        return Event(time, state, self.origin, self.frame)

    ######################################################################################
    # Class Method
    ######################################################################################

    @staticmethod
    def resolution(dpos_duv):
        """The spatial resolution on a surface.

        Parameters:
            dpos_duv (Vector3): A Vector3 with denominator shape (2,), defining the
                partial derivatives `d(x,y,z)/d(u,v)`, where `(x,y,z)` are the 3-D
                coordinates of a point on the surface and `(u,v)` are pixel coordinates.

        Returns:
            tuple[Scalar, Scalar]: `(res_min, res_max)`, where `res_min` contains
            resolution values (km/pixel) in the direction of finest spatial resolution
            and `res_max` contains resolution values (km/pixel) in the direction of
            coarsest spatial resolution.

        Notes:
            For the best solution, the derivatives should be adjusted such that the u-axis
            and the v-axis are locally perpendicular. See the source code of
            Backplane.dlos_duv1 in backplane/__init__.py for details.
        """

        return Surface._resolution(dpos_duv)

    @staticmethod
    def _resolution(dpos_duv, _unittest=False):
        """The spatial resolution on a surface.

        Parameters:
            dpos_duv (Vector3): A Vector3 with denominator shape (2,), defining the
                partial derivatives `d(x,y,z)/d(u,v)`, where `(x,y,z)` are the 3-D
                coordinates of a point on the surface and `(u,v)` are pixel coordinates.
            _unittest (bool, optional): True to return `(dpos_du_prime, dpos_dv_prime)`
                instead of `(res_min, res_max)`. This supports unit tests to confirm that
                the dot product of these two vectors is small.

        Returns:
            tuple[Scalar, Scalar]: `(res_min, res_max)`, where `res_min` contains
            resolution values (km/pixel) in the direction of finest spatial resolution
            and `res_max` contains resolution values (km/pixel) in the direction of
            coarsest spatial resolution.
        """

        # Define vectors parallel to the surface, containing the derivatives
        # with respect to each pixel coordinate.
        (dpos_du, dpos_dv) = dpos_duv.extract_denoms()

        # The resolution should be independent of the rotation angle of the
        # grid. We therefore need to solve for an angle theta such that
        #   dpos_du' = cos(theta) dpos_du - sin(theta) dpos_dv
        #   dpos_dv' = sin(theta) dpos_du + cos(theta) dpos_dv
        # where
        #   dpos_du' <dot> dpos_dv' = 0
        #
        # Then, the magnitudes of dpos_du' and dpos_dv' will be the local values
        # of finest and coarsest spatial resolution (in either order).
        #
        # Let t = tan(theta):
        #   dpos_du' ~   dpos_du - t dpos_dv
        #   dpos_dv' ~ t dpos_du +   dpos_dv
        # subject to the requirement that the dot product is zero.
        #
        # 0 =   t^2 (dpos_du <dot> dpos_dv)
        #     + t   (|dpos_dv|^2 - |dpos_du|^2)
        #     -     (dpos_du <dot> dpos_dv)
        #
        # Use the quadratic formula.

        a = dpos_du.dot(dpos_dv)
        b = dpos_dv.dot(dpos_dv) - dpos_du.dot(dpos_du)
        # c = -a    # not actually needed

        # discr = b**2 - 4*a*c
        discr = b**2 + 4*a**2

        # There are two solutions, for which theta differs by pi/2 as one would
        # expect. For our purposes, the highest-precision formulation is:
        #   t = -2*c / (b + sign(b) * sqrt(discr))
        # because:
        # 1. b and sqrt(discr) could be close, making subtraction imprecise.
        # 2. a could be close to zero, so we don't want to divide by 2*a.

        t = (2*a) / (b + b.sign() * discr.sqrt())

        # Now infer the cosine and sine and construct the primed partials
        cos_theta = 1. / (1 + t**2).sqrt()
        sin_theta = t * cos_theta

        dpos_du_prime = (cos_theta * dpos_du - sin_theta * dpos_dv)
        dpos_dv_prime = (sin_theta * dpos_du + cos_theta * dpos_dv)

        # For purposes of testing, let's make sure the dot product is small
        if _unittest:
            return (dpos_du_prime, dpos_dv_prime)

        # Return the minima and maxima separately
        dpos_du_prime_norm = dpos_du_prime.norm()
        dpos_dv_prime_norm = dpos_dv_prime.norm()

        minres = Scalar.minimum(dpos_du_prime_norm, dpos_dv_prime_norm)
        maxres = Scalar.maximum(dpos_du_prime_norm, dpos_dv_prime_norm)

        return (minres, maxres)

##########################################################################################

import oops.surface._photon_solver as _photon_solver
Surface.photon_to_event        = _photon_solver.photon_to_event
Surface.photon_from_event      = _photon_solver.photon_from_event
Surface.photon_to_coords       = _photon_solver.photon_to_coords
Surface.photon_from_coords     = _photon_solver.photon_from_coords
Surface.photon_normal_to_event = _photon_solver.photon_normal_to_event
Surface.photon_event_to_normal = _photon_solver.photon_event_to_normal
Surface.photon_normal_to_path  = _photon_solver.photon_normal_to_path
Surface.photon_path_to_normal  = _photon_solver.photon_path_to_normal

# CoordPath._solve_photon reaches this through the Surface instance
Surface._solve_photon_by_coords = _photon_solver._solve_photon_by_coords

##########################################################################################
