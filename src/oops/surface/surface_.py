##########################################################################################
# oops/surface/surface_.py
##########################################################################################

import numbers
import numpy as np

from polymath          import Boolean, Scalar, Vector3
from oops.event        import Event
from oops.mutable      import Mutable


class Surface(Mutable):
    """An abstract class describing a 2-D object that moves and rotates in space.

    A surface employs an internal coordinate system, not necessarily rectangular, in which
    two primary coordinates define locations on the surface, and an optional third
    coordinate can define points above or below that surface.

    Properties:
        origin (Path): The waypoint of the path defining the surface's center.
        frame (Frame): The wayframe of the frame in which the surface is defined.
        unmasked (Surface): An unmasked version of this surface. If the surface has no
            mask, this returns self.
        intercept_key (tuple): A unique, immutable key that defines the surface. Some
            surface classes are identical except for a mask or coordinate definition;
            those classes return the same intercept key.
    """

    # Class constant to avoid circular references
    _Body = None            # filled in by body.py

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = True
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = True
    intercept_DERIVS_ARE_IMPLEMENTED = True
    normal_DERIVS_ARE_IMPLEMENTED = True
    intercept_with_normal_DERIVS_ARE_IMPLEMENTED = True
    intercept_normal_to_DERIVS_ARE_IMPLEMENTED = True

    # Default properties; override as needed

    # A virtual path is one whose 3-D shape depends on the position of the
    # observer. For example, the "ansa" surface is virtual, because it is
    # defined as a locus of points where the line of sight to the observer are
    # perpendicular to the direction to the ring's rotation pole.
    IS_VIRTUAL = False

    # A time-dependent path is one whose 3-D shape varies with time.
    IS_TIME_DEPENDENT = False

    # True for any surface that has an interior
    HAS_INTERIOR = False

    def _coords_from_vector3_check(self, axes):
        """Validate axes as equal to 2 or 3."""

        if not isinstance(axes, numbers.Integral) or axes not in (2, 3):
            raise ValueError(f'axes must be 2 or 3 in {type(self).__name__}'
                             '.coords_from_vector3()')

    def _vector3_from_coords_check(self, coords):
        """Validate coords as a tuple of 2 or 3 Scalars."""

        if not isinstance(coords, (tuple, list)):
            raise TypeError('invalid coords in %s.vector3_from_coords(): '
                            'class %s given; list or tuple required'
                            % (type(self).__name__, type(coords).__name__))

        if len(coords) not in (2, 3):
            raise ValueError(f'2 or 3 coords required in {type(self).__name__}'
                             '.vector3_from_coords()')

    ######################################################################################
    # Each subclass must override...
    ######################################################################################

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the surface, relative to this surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this surface's origin
                and frame. Ignored for solid surfaces but needed for virtual surfaces.
            time (Scalar, optional): Time at which to evaluate the surface; ignored unless
                the surface is time-variable.
            axes (int, optional): 2 or 3, indicating whether to return a tuple of two or
                three Scalar objects.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (optional): Optional data used to expedite this calculation. The
                specific meaning depends on the Surface subclass.

        Returns:
            tuple[Scalar, ...]: Two or three coordinate values, depending on the input
                value of axes.
        """

        raise NotImplementedError(f'{type(self).__name__}.coords_from_vector3() is not '
                                  'implemented')

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False):
        """The position where a point with the given coordinates falls relative to this
        surface's origin and frame.

        Parameters:
            coords (tuple): Two or three Scalars defining coordinates at or near this
                surface. These can have different shapes, but must be broadcastable to a
                common shape.
            obs (Vector3, optional): Observer position relative to this surface's origin
                and frame. Ignored for solid surfaces but needed for virtual surfaces.
            time (Scalar, optional): Time at which to evaluate the surface; ignored unless
                the surface is time-variable.
            derivs (bool, optional): True to propagate any derivatives inside the
                coordinates and obs into the returned position vectors.

        Returns:
            Vector3: Intercept points defined by the coordinates.
        """

        raise NotImplementedError(f'{type(self).__name__}.vector3_from_coords() is not '
                                  'implemented')

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False, guess=None,
                  hints=None):
        """The position where a specified line of sight intercepts the surface.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this surface's frame.
            time (Scalar, optional): Time at the surface; ignored unless the surface is
                time-variable.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface. Needed for closed surfaces that
                have two intercept points; ignored otherwise.
            derivs (bool, optional): True to propagate any derivatives inside obs and los
                into the returned intercept point.
            guess (Scalar, optional): Optional initial guess at the coefficient t such
                that: intercept = obs + t * los.
            hints (optional): Any data that might be useful to carry over from one call
                to the next. If not None, hint values are appended to the returned tuple.
                Use hints=True if you lack an initial value but require the new value to
                be returned.

        Returns:
            tuple[Vector3, Scalar[, hints]]: (pos, t) or (pos, t, hints), where:

            * `pos` (Vector3): Intercept points on the surface, in km.
            * `t` (Scalar): Such that: intercept = obs + t * los.
            * `hints`: Latest version of any hint values; included if the input value
              of hints is not None (the default). Note that some Surface subclasses do not
              use hints; for these, the input value of the hints is itself returned if it
              is not None.
        """

        raise NotImplementedError(f'{type(self).__name__}.intercept() is not implemented')

    def normal(self, pos, *, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Parameters:
            pos (Vector3): Positions at or near the surface relative to this surface's
                origin and frame.
            time (Scalar, optional): Time at the surface; ignored unless the surface is
                time-variable.
            derivs (bool, optional): True to propagate any derivatives of pos into the
                returned normal vectors.

        Returns:
            Vector3: Directions normal to the surface that pass through the position.
                Lengths are arbitrary.
        """

        raise NotImplementedError(f'{type(self).__name__}.normal() is not implemented')

    ######################################################################################
    # Optional Methods...
    ######################################################################################

    def intercept_with_normal(self, normal, *, time=None, derivs=False):
        """Surface point where the outward normal vector parallels the given vector.

        Parameters:
            normal (Vector3): Normal vectors in the surface's frame.
            time (Scalar, optional): Time at the surface; ignored unless the surface is
                time-variable.
            derivs (bool, optional): True to propagate derivatives in the normal vector
                into the returned surface points.

        Returns:
            Vector3: Surface intercept points, in km. Where no solution exists, the
                returned Vector3 will be masked.
        """

        raise NotImplementedError(f'{type(self).__name__}.intercept_with_normal() is not '
                                  'implemented')

    def intercept_normal_to(self, pos, *, time=None, direction='dep', derivs=False,
                            guess=None):
        """Surface point whose normal vector passes through a given position.

        This function can have multiple values, in which case the nearest of the surface
        points should be the one returned.

        Parameters:
            pos (Vector3): Positions at or near the surface relative to this surface's
                origin and frame.
            time (Scalar, optional): Time at the surface; ignored unless the surface is
                time-variable.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface. Needed for closed surfaces that
                have two intercept points; ignored otherwise.
            derivs (bool, optional): True to propagate derivatives in pos into the
                returned intercepts.
            guess (Scalar, optional): Optional initial guess a coefficient array p such
                that: intercept = pos + p * normal(intercept) If provided, the converged
                value of p is included in the returned results; use guess=True to include
                this in the return even if an initial guess is not available.

        Returns:
            Intercept or (intercept, p), where:

            * `intercept` (Vector3): A vector3 of surface intercept points, in km. Where
              no solution exists, the returned vector will be masked.
            * `p` (Scalar): The converged solution such that intercept = pos + p *
              normal(intercept); included if guess is not None. Note that some Surface
              subclasses do not use an initial guess; for these, the input value of the
              guess is itself returned as p (if it is not None).
        """

        raise NotImplementedError(f'{type(self).__name__}.intercept_normal_to() is not '
                                  'implemented')

    def velocity(self, pos, *, time=None):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or local wind
        speeds on a planet.

        Parameters:
            pos (Vector3): Positions at or near the surface relative to this surface's
                origin and frame.
            time (Scalar, optional): Time at the surface; ignored unless the surface is
                time-variable.

        Returns:
            Vector3: Velocities, in units of km/s.
        """

        return Vector3.ZERO

    def position_is_inside(self, pos, *, obs=None, time=None):
        """Where positions are inside the surface.

        Parameters:
            pos (Vector3): Positions at or near the surface relative to this surface's
                origin and frame.
            obs (Vector3, optional): Observer positions. Ignored for solid surfaces but
                needed for virtual surfaces.
            time (Scalar, optional): Time at which to evaluate the surface; ignored unless
                the surface is time-variable.

        Returns:
            Boolean: Boolean True where positions are inside the surface.
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
        """The reference surface for this one."""

        return self     # default is to return self

    ######################################################################################
    # Event-coordinate conversions. Generally should not require overrides
    ######################################################################################

    def coords_of_event(self, event, *, obs=None, axes=3, derivs=False):
        """Coordinate values associated with an event near the surface.

        Parameters:
            event (Event): An event occurring at or near the surface.
            obs (Event, optional): Observing event, which may occur at a different time.
            axes (int, optional): 2 or 3, indicating whether to return a tuple of two or
                three Scalar objects.
            derivs (bool, optional): If True, then all derivatives are carried forward
                into the event; if False, only time derivatives are included.

        Returns:
            Coordinate values packaged as a tuple containing two or three unitless
                Scalars, one for each coordinate.
        """

        # Locate the events WRT the surface frame
        cept_in_frame = event.wrt(self.origin, self.frame, derivs=derivs).state

        if obs is not None:
            obs_in_frame = obs.wrt(self.origin, self.frame, derivs=derivs).state
        else:
            obs_in_frame = None

        # Evaluate the coords and optional derivatives
        hints = event.hints if hasattr(event, 'hints') else None
        return self.coords_from_vector3(cept_in_frame, obs=obs_in_frame, time=event.time,
                                        axes=axes, derivs=True, hints=hints)

    def apply_coords_to_event(self, event, *, obs=None, axes=3, derivs=True):
        """A shallow copy of this event with attributes coord1, coord2, coord3 added,
        along with any mask.

        Parameters:
            event (Event): An event occurring at or near the surface.
            obs (Event, optional): Observing event, which may occur at a different time.
            axes (int, optional): 2 or 3, indicating whether to return a tuple of two or
                three Scalar objects.
            derivs (bool, optional): If True, then all derivatives are carried forward
                into the event; if False, only time derivatives are included.

        Returns:
            Clone of event with new attributes coord1, coord2, coord3.
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
        """Converts a time and coordinates in the surface's internal coordinate system
        into an event object.

        Parameters:
            time (Scalar): Time in seconds TDB.
            coords (tuple[Scalar, ...]): 2 or 3 coordinates.
            obs (Vector3, optional): Observer position relative to this surface's origin
                and frame. Ignored for solid surfaces but needed for virtual surfaces.
            derivs (bool, optional): If True, then all derivatives are carried forward
                into the event; if False, only time derivatives are included.

        Returns:
            Event: An event object relative to the origin and frame of the surface.
        """

        # Interpret coords
        if len(coords) == 2:
            (coord1, coord2) = coords
            coord3 = Scalar.ZERO
        else:
            (coord1, coord2, coord3) = coords

        # Strip derivatives is necessary, but not d_dt
        if not derivs:
            coord1 = coord1.without_derivs(preserve='t')
            coord2 = coord2.without_derivs(preserve='t')
            coord3 = coord3.without_derivs(preserve='t')

            if obs is not None:
                obs = obs.without_derivs(preserve='t')

        # Determine position and velocity
        state = self.vector3_from_coords((coord1, coord2, coord3), obs=obs,
                                         time=time, derivs=True)

        # Return the event
        return Event(time, state, self.origin, self.frame)

    ######################################################################################
    # Class Method
    ######################################################################################

    @staticmethod
    def resolution(dpos_duv, _unittest=False):
        """Determine the spatial resolution on a surface.

        Parameters:
            dpos_duv (Vector3): A Vector3 with denominator shape (2,), defining the
                partial derivatives d(x,y,z)/d(u,v), where (x,y,z) are the 3-D coordinates
                of a point on the surface, and (u,v) are pixel coordinates.

        Returns:
            (tuple): A tuple (res_min, res_max) where:

            * `res_min` (Scalar): A Scalar containing resolution values (km/pixel) in the
              direction of finest spatial resolution.
            * `res_max` (Scalar): A Scalar containing resolution values (km/pixel) in the
              direction of coarsest spatial resolution.

        Notes:
            For the best solution, the derivatives should be adjusted such that the u-axis
            and the v-axis are locally perpendicular. See the source code of
            Backplane.dlos_duv1 in backplane/__init__.py for details.
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

from ._photon_solver import (photon_from_event, photon_to_event,
                             photon_from_event_by_coords, photon_to_event_by_coords,
                             photon_normal_to_event, photon_event_to_normal,
                             photon_normal_to_path, photon_path_to_normal,
                             _fully_masked_result, _solve_photon_by_los,
                             _solve_photon_by_coords, _solve_photon_normal_to_surface,
                             _solve_normal_for_photon_event)
Surface.photon_from_event           = photon_from_event
Surface.photon_to_event             = photon_to_event
Surface.photon_from_event_by_coords = photon_from_event_by_coords
Surface.photon_to_event_by_coords   = photon_to_event_by_coords
Surface.photon_normal_to_event      = photon_normal_to_event
Surface.photon_event_to_normal      = photon_event_to_normal
Surface.photon_normal_to_path       = photon_normal_to_path
Surface.photon_path_to_normal       = photon_path_to_normal

Surface._fully_masked_result              = _fully_masked_result
Surface._solve_photon_by_los              = _solve_photon_by_los
Surface._solve_photon_by_coords           = _solve_photon_by_coords
Surface._solve_photon_normal_to_surface   = _solve_photon_normal_to_surface
Surface._solve_normal_for_photon_event    = _solve_normal_for_photon_event

##########################################################################################
