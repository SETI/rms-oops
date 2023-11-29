################################################################################
# oops/surface/__init__.py: Abstract class Surface
################################################################################

import numbers
import numpy as np

from polymath       import Boolean, Qube, Scalar, Vector3
from oops.config    import SURFACE_PHOTONS, LOGGING
from oops.constants import C
from oops.event     import Event
from oops.frame     import Frame
from oops.path      import Path

class Surface(object):
    """Surface is an abstract class describing a 2-D object that moves and
    rotates in space. A surface employs an internal coordinate system, not
    necessarily rectangular, in which two primary coordinates define locations
    on the surface, and an optional third coordinate can define points above or
    below that surface.

    Required attributes:
        origin          the waypoint of the path defining the surface's center.
        frame           the wayframe of the frame in which the surface is
                        defined.
        unmasked        an un-masked version of this surface. If the surface has
                        no mask, this returns self.
        intercept_key   a unique, immutable key that defines the surface. Note
                        that some surface classes are identical except for a
                        mask or coordinate definition; these classes should
                        return the same intercept key.
    """

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

    DEBUG = False           # True to log iteration convergence steps

    ########################################
    # Each subclass must override...
    ########################################

    def __init__(self):
        """Constructor for a Surface object."""

        pass

    #===========================================================================
    def _coords_from_vector3_check(self, axes):
        """Validate axes as equal to 2 or 3."""

        if not isinstance(axes, numbers.Integral):
            raise TypeError('invalid axes in %s.vector3_from_coords(): '
                            'class %s given; int required'
                            % (type(self).__name__, type(axes).__name__))

        if axes not in (2, 3):
            raise ValueError('invalid axes in %s.coords_from_vector3(): %s; '
                             'must be 2 or 3'
                             % (type(self).__name__, axes))

    #===========================================================================
    def _vector3_from_coords_check(self, coords):
        """Validate coords as a tuple of 2 or 3 Scalars."""

        if not isinstance(coords, (tuple, list)):
            raise TypeError('invalid coords in %s.vector3_from_coords(): '
                            'class %s given; list or tuple required'
                            % (type(self).__name__, type(coords).__name__))

        if len(coords) not in (2, 3):
            raise ValueError('invalid coords in %s.vector3_from_coords(): '
                             '%d given; 2 or 3 required'
                             % (type(self).__name__, len(coords)))

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None):
        """Surface coordinates associated with a position vector.

        Input:
            pos         a Vector3 of positions at or near the surface, relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame. Ignored for solid surfaces
                        but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored
                        unless the surface is time-variable.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       optional data used to expedite this calculation. The
                        specific meaning depends on the Surface subclass.

        Return:
            coords      a tuple of two or three coordinate values, depending on
                        the input value of axes.
        """

        raise NotImplementedError('%s.coords_from_vector3() is not implemented'
                                  % type(self).__name__)

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False):
        """The position where a point with the given coordinates falls relative
        to this surface's origin and frame.

        Input:
            coords      a tuple of two or three Scalars defining coordinates at
                        or near this surface. These can have different shapes,
                        but must be broadcastable to a common shape.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame. Ignored for solid surfaces
                        but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored
                        unless the surface is time-variable.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.
        """

        raise NotImplementedError('%s.vector3_from_coords() is not implemented'
                                  % type(self).__name__)

    #===========================================================================
    def intercept(self, obs, los, time=None, direction='dep', derivs=False,
                                  guess=None, hints=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3 relative to this
                        surface's origin and frame.
            los         line of sight as a Vector3 in this surface's frame.
            time        a Scalar time at the surface; ignored unless the surface
                        is time-variable.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface. Needed for closed
                        surfaces that have two intercept points; ignored
                        otherwise.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los
            hints       any data that might be useful to carry over from one
                        call to the next. If not None, hint values are appended
                        to the returned tuple. Use hints=True if you lack an
                        initial value but require the new value to be returned.

        Return:         a tuple (pos, t) or (pos, t, hints), where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
            hints       latest version of any hint values; included if the input
                        value of hints is not None (the default).

                        Note that some Surface subclasses do not use hints; for
                        these, the input value of the hints is itself returned
                        if it is not None.
        """

        raise NotImplementedError('%s.intercept() is not implemented'
                                  % type(self).__name__)

    #===========================================================================
    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at the surface; ignored unless the surface
                        is time-variable.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        raise NotImplementedError('%s.normal() is not implemented'
                                  % type(self).__name__)

    ########################################
    # Optional Methods...
    ########################################

    def intercept_with_normal(self, normal, time=None, derivs=False):
        """Surface point where the outward normal vector parallels the given
        vector.

        Input:
            normal      a Vector3 of normal vectors in the surface's frame.
            time        a Scalar time at the surface; ignored unless the surface
                        is time-variable.
            derivs      True to propagate derivatives in the normal vector into
                        the returned surface points.

        Return:         a Vector3 of surface intercept points, in km. Where no
                        solution exists, the returned Vector3 will be masked.
        """

        raise NotImplementedError('%s.intercept_with_normal() is not '
                                  'implemented' % type(self).__name__)

    #===========================================================================
    def intercept_normal_to(self, pos, time=None, direction='dep', derivs=False,
                                       guess=None):
        """Surface point whose normal vector passes through a given position.

        This function can have multiple values, in which case the nearest of the
        surface points should be the one returned.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at the surface; ignored unless the surface
                        is time-variable.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface. Needed for closed
                        surfaces that have two intercept points; ignored
                        otherwise.
            derivs      True to propagate derivatives in pos into the returned
                        intercepts.
            guess       optional initial guess a coefficient array p such that:
                            intercept = pos + p * normal(intercept)
                        If provided, the converged value of p is included in the
                        returned results; use guess=True to include this in the
                        return even if an initial guess is not available.

        Return:         intercept or (intercept, p), where
            intercept   a vector3 of surface intercept points, in km. Where no
                        solution exists, the returned vector will be masked.
            p           the converged solution such that
                            intercept = pos + p * normal(intercept);
                        included if guess is not None.

                        Note that some Surface subclasses do not use an initial
                        guess; for these, the input value of the guess is itself
                        returned as p (if it is not None).
        """

        raise NotImplementedError('%s.intercept_normal_to() is not implemented'
                                  % type(self).__name__)

    #===========================================================================
    def velocity(self, pos, time=None):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at the surface; ignored unless the surface
                        is time-variable.

        Return:         a Vector3 of velocities, in units of km/s.
        """

        return Vector3.ZERO

    #===========================================================================
    def position_is_inside(self, pos, obs=None, time=None):
        """Where positions are inside the surface.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored
                        unless the surface is time-variable.

        Return:         Boolean True where positions are inside the surface
        """

        if self.HAS_INTERIOR:
            raise NotImplementedError('%s.position_is_inside() is not '
                                      'implemented' % type(self).__name__)

        return Boolean.FALSE

    ############################################################################
    # Support for surfaces derived from other surfaces. E.g., surfaces using
    # different coordinates or with boundaries applied.
    ############################################################################

    def reference(self):
        """The reference surface for this one."""

        return self     # default is to return self

    ############################################################################
    # Event-coordinate conversions. Generally should not require overrides
    ############################################################################

    def coords_of_event(self, event, obs=None, axes=3, derivs=False):
        """Coordinate values associated with an event near the surface.

        Input:
            event       an event occurring at or near the surface.
            obs         observing event, which may occur at a different time.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      If True, then all derivatives are carried forward into
                        the event; if False, only time derivatives are included.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.
        """

        # Locate the events WRT the surface frame
        cept_in_frame = event.wrt(self.origin, self.frame,
                                  derivs=derivs).state

        if obs is not None:
            obs_in_frame = obs.wrt(self.origin, self.frame,
                                      derivs=derivs).state
        else:
            obs_in_frame = None

        # Evaluate the coords and optional derivatives
        hints = event.hints if hasattr(event, 'hints') else None
        return self.coords_from_vector3(cept_in_frame, obs=obs_in_frame,
                                        time=event.time, axes=axes,
                                        derivs=True, hints=hints)

    #===========================================================================
    def apply_coords_to_event(self, event, obs=None, axes=3, derivs=True):
        """A shallow copy of this event with attributes coord1, coord2, coord3
        added, along with any mask.

        Input:
            event       an event occurring at or near the surface.
            obs         observing event, which may occur at a different time.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      If True, then all derivatives are carried forward into
                        the event; if False, only time derivatives are included.

        Return:         clone of event with new attributes coord1, coord2,
                        coord3.
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

    #===========================================================================
    def event_at_coords(self, time, coords, obs=None, derivs=False):
        """Converts a time and coordinates in the surface's internal coordinate
        system into an event object.

        Input:
            time        the Scalar of time values at which to evaluate the
                        coordinates.
            coords      a tuple containing the two or three coordinate values.
                        If only two are provided, the third is assumed to be
                        zero.
            obs         a Vector3 of observer positions. Needed for virtual
                        surfaces; can be None otherwise.
            derivs      If True, then all derivatives are carried forward into
                        the event; if False, only time derivatives are included.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:         an event object relative to the origin and frame of the
                        surface.
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

    ############################################################################
    # Photon Solver based on line of sight
    ############################################################################

    def photon_to_event(self, arrival, derivs=False, guess=None, antimask=None,
                              quick={}, converge={}):
        """Photon departure from this surface, given arrival and line of sight.

        See _solve_photon_by_los() for details.
        """

        return self._solve_photon_by_los(arrival, -1, derivs, guess, antimask,
                                                      quick, converge)

    #===========================================================================
    def photon_from_event(self, departure, derivs=False, guess=None,
                                antimask=None, quick={}, converge={}):
        """Photon arrival at this surface, given departure and line of sight.

        See _solve_photon_by_los() for details.
        """

        return self._solve_photon_by_los(departure, +1, derivs, guess, antimask,
                                                        quick, converge)

    #===========================================================================
    def _solve_photon_by_los(self, link, sign, derivs=False, guess=None,
                                   antimask=None, quick={}, converge={}):
        """Solve for a photon surface intercept from event and line of sight.

        Input:
            link        the link event of a photon's arrival or departure.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the surface and arriving later at the
                           link.
                        +1 to return later events, corresponding to photons
                           departing from the link and arriving later at the
                           surface.

            derivs      True to propagate derivatives of the link position and
                        and line of sight into the returned event. Derivatives
                        with respect to time are always retained.

            guess       an initial guess to use as the event time for the
                        surface; otherwise None. Should be used if the event
                        time was already returned from a similar calculation.

            antimask    if not None, this is a boolean array to be applied to
                        event times and positions. Only the indices where
                        antimask=True will be used in the solution.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py/SURFACE_PHOTONS.

        Return:         a tuple (surface_event, link_event).

            surface_event
                        the event on the surface that matches the light travel
                        time from the link event. This event is defined in the
                        frame of the surface and relative to the surface's
                        origin.

                        The surface event also contains three Scalar subfields,
                        "coord1", "coord2", and "coord3", containing the surface
                        coordinates at the intercept point (and their optional
                        derivatives).

            link_event  a copy of the given event, with the photon travel time
                        filled in.

            If sign is +1, then these subfields and derivatives are defined.
                In surface_event:
                    arr         direction of the arriving photon at the surface.
                    arr_lt      (negative) light travel time from the link
                                event to the surface.
                In link_event:
                    dep_lt      light travel time between the events.

            If sign is -1, then 'arr' and 'dep' are swapped for the two events.
            Note that subfield 'arr_lt' is always negative and 'dep_lt' is
            always positive. Subfields 'arr' and 'dep' have the same direction
            in both events.

        Convergence parameters are as follows:
            max_iterations  the maximum number of iterations of Newton's method
                            to perform. It should almost never need to be > 6.
            dlt_precision   iteration stops when the largest change in light
                            travel time between one iteration and the next falls
                            below this threshold (in seconds).
            dlt_limit       the maximum allowed absolute value of the change in
                            light travel time from the nominal range calculated
                            initially. Changes in light travel with absolute
                            values larger than this limit are clipped. This
                            prevents the divergence of the solution in some
                            cases.
        """

        # Hide link derivative here; we will restore them at the end
        if derivs:
            link_with_derivs = link
            link = link.wod
        else:
            link = link.wod     # preserves time-derivatives; removes others
            link_with_derivs = link

        # Assemble convergence parameters
        if converge:
            defaults = SURFACE_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = SURFACE_PHOTONS.__dict__

        iters = converge['max_iterations']
        precision = converge['dlt_precision']
        limit = converge['dlt_limit']

        # Interpret the quick parameters
        if isinstance(quick, dict):
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Interpret the sign
        if sign < 0:        # light time < 0, photon from surface to observer
            signed_c = -C
            surface_key = 'dep'
            link_key = 'arr'
        else:               # light time > 0, photon from observer to surface
            signed_c = C
            link_key = 'dep'
            surface_key = 'arr'

        # Define the antimask
        if antimask is None:
            antimask = link.antimask
        else:
            antimask = antimask & link.antimask

        # If the link is entirely masked...
        if not np.any(antimask):
            return self._fully_masked_result(link_with_derivs, link_key,
                                             coords=True)

        # Shrink the event
        original_shape = link.shape
        link = link.shrink(antimask)

        # Define quantities with respect to SSB in J2000
        link_wrt_ssb = link.wrt_ssb(derivs=False, quick=quick)
        path_wrt_ssb = self.origin.wrt(Path.SSB, Frame.J2000)
        frame_wrt_j2000 = self.frame.wrt(Frame.J2000)

        # Prepare for iteration
        obs_wrt_ssb = link_wrt_ssb.pos
        los_in_j2000 = link_wrt_ssb.get_subfield(link_key).wod.with_norm(C)

        # Validate the guess input
        if guess is not None:
            guess = Scalar.as_scalar(guess, recursive=False).wod
            guess = guess.shrink(antimask)

            # Masked values in the guess are not usable
            if np.all(guess.mask):
                guess = None
            elif np.any(guess.mask):
                guess = guess.copy()
                guess[guess.mask] = guess.mean()

        # Prepare the first guesses at the surface_time and lt
        if guess is None:
            # If no guess was provided, base the time on the range to the origin
            origin_event = path_wrt_ssb.event_at_time(link.time, quick=quick)
            lt = (origin_event.pos.wod - obs_wrt_ssb.wod).norm() / signed_c
            surface_time = link.time + lt
        else:
            surface_time = guess
            lt = surface_time - link.time.wod

        # Set light travel time limits to avoid a diverging solution
        lt_min = lt.min(builtins=True) - limit
        lt_max = lt.max(builtins=True) + limit

        # Iterate to solve for lt and surface time. Convergence is rapid because
        # all speeds are non-relativistic.
        max_dlt = np.inf
        converged = False
        hints = True                    # speeds up some calculations
        for count in range(iters):

            # Quicken the path and frame as soon as the range of surface times
            # indicates that this would be beneficial.
            path_wrt_ssb = path_wrt_ssb.quick_path(surface_time, quick=quick)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quick)
                # Below, we specify quick=False because the path and frame are
                # already quickened.

            # Locate the intercept points relative to the origin in SSB/J2000,
            # using the current surface time
            origin_wrt_ssb = path_wrt_ssb.event_at_time(surface_time,
                                                        quick=False).pos
            cept_in_j2000 = (obs_wrt_ssb - origin_wrt_ssb) + lt * los_in_j2000

            # Rotate into the surface-fixed frame
            surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                              quick=False)
            cept_in_frame = surface_xform.rotate(cept_in_j2000, derivs=False)
            los_in_frame = surface_xform.rotate(los_in_j2000, derivs=False)

            # Update the intercept time via a shift along the line of sight
            (cept_in_frame, dlt, hints) = self.intercept(cept_in_frame,
                                                         los_in_frame,
                                                         time=surface_time,
                                                         direction=surface_key,
                                                         derivs=False,
                                                         hints=hints)
            new_lt = lt + dlt

            # Clip time
            new_lt = new_lt.clip(lt_min, lt_max, remask=False)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Surface.DEBUG:
                LOGGING.convergence('%s._solve_photon_by_los(): '
                                    'iter=%d; change[s]=%.6g'
                                    % (type(self).__name__, count+1,
                                       max(max_dlt, 0.)))

            if max_dlt <= precision:        # converged or fully masked
                converged = True
                break

            if max_dlt >= prev_max_dlt:     # failure to converge
                break

            # Re-evaluate the surface time
            surface_time = link.time + lt

        #### END OF LOOP

        if not converged:
            LOGGING.warn('Surface._solve_photon_by_los did not converge;',
                         'iter=%d; change=%.6g' % (count+1, max_dlt))

        # One last iteration with derivatives included
        surface_time = link.time + lt

        if link is not link_with_derivs:
            link = link_with_derivs
            link = link.shrink(antimask)
            link_wrt_ssb = link.wrt_ssb(derivs=True, quick=quick)

        obs_wrt_ssb = link_wrt_ssb.state
        los_in_j2000 = link_wrt_ssb.get_subfield(link_key).with_norm(C)

        origin_wrt_ssb = path_wrt_ssb.event_at_time(surface_time,
                                                    quick=False).state
        cept_in_j2000 = (obs_wrt_ssb - origin_wrt_ssb) + lt * los_in_j2000

        surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                          quick=False)
        cept_in_frame = surface_xform.rotate(cept_in_j2000, derivs=True)
        los_in_frame = surface_xform.rotate(los_in_j2000, derivs=True)

        (cept_in_frame, dlt, hints) = self.intercept(cept_in_frame,
                                                     los_in_frame,
                                                     time=surface_time,
                                                     direction=surface_key,
                                                     derivs=True,
                                                     hints=hints)
        new_lt = lt + dlt
        lt = new_lt.clip(lt_min, lt_max, remask=False)
        surface_time = link.time + lt

        # Update the mask on light time to hide intercepts outside the defined
        # limits
        new_mask = ((lt.values * sign < 0.)
                    | (lt.values == lt_min)
                    | (lt.values == lt_max))
        if np.any(new_mask):
            lt = lt.remask_or(new_mask)

        # If the link is entirely masked, return masked results
        if max_dlt < 0. or np.all(surface_time.mask):
            return self._fully_masked_result(link_with_derivs, link_key,
                                             coords=True)

        #### Create the surface event in its own frame

        # The intercept event with respect to the surface has a time-derivative
        # due to the rate of change of the line of sight. However, THIS IS NOT A
        # PHYSICAL VELOCITY. To define the surface event properly, we need to
        # remove the time derivative of cept_wrt_surface. We assign it a new
        # name d_dT to distinguish it from d_dt.

        event_state = cept_in_frame.rename_deriv('t', 'T', method='add')
        event_time  = surface_time.rename_deriv('t', 'T', method='add')
        surface_event = Event(event_time, event_state, self.origin, self.frame)

        # Subfields are calculated using the original cept_in_frame, so these
        # attributes will have correct time-derivatives. This is OK because
        # these time-derivatives are not physical velocities.

        perp = self.normal(cept_in_frame, time=surface_time, derivs=True)
        vflat = self.velocity(cept_in_frame, time=surface_time)
        surface_event.insert_subfield('perp', perp)
        surface_event.insert_subfield('vflat', vflat)
        surface_event.insert_subfield(surface_key, los_in_frame.unit())
        surface_event.insert_subfield(surface_key + '_lt', -lt)

        # Fill in coordinate subfields
        obs_in_frame = cept_in_frame - lt * los_in_frame
        coords = self.coords_from_vector3(cept_in_frame, obs_in_frame,
                                          time=surface_time, axes=3,
                                          derivs=True, hints=hints)
        surface_event.insert_subfield('coord1', coords[0])
        surface_event.insert_subfield('coord2', coords[1])
        surface_event.insert_subfield('coord3', coords[2])

        # Save the hints if any
        if hints is not True:
            surface_event.insert_subfield('hints', hints)

        # Construct the updated link_event
        new_link = link.replace(link_key + '_lt', lt)

        # Unshrink
        surface_event = surface_event.unshrink(antimask, shape=original_shape)
        new_link = new_link.unshrink(antimask, shape=original_shape)

        return (surface_event, new_link)

    def _fully_masked_result(self, link, link_key, coords=False):
        """Internal function to return an entirely masked result."""

        # Identify derivatives in the link event
        deriv_denoms = {}
        for thing in link.__dict__.values():
            if isinstance(thing, Qube):
                for key, deriv in thing.derivs.items():
                    deriv_denoms[key] = deriv.denom

        # Create empty Vector3 and Scalar
        vector = Vector3.ones(link.shape, mask=True)
        scalar = Scalar.zeros(link.shape, mask=True)

        # Insert all the derivs
        for key, denom in deriv_denoms.items():
            vector.insert_deriv(key, Vector3.ones(link.shape, denom=denom,
                                                  mask=True))
            scalar.insert_deriv(key, Scalar.ones(link.shape, denom=denom,
                                                 mask=True))

        # Add link key attributes for the new, masked link
        new_link = link.all_masked()
        new_link = new_link.replace(link_key, vector,
                                    link_key + '_lt', scalar)

        # Create the surface event
        surface_key = 'arr' if link_key == 'dep' else 'dep'
        surface_event = Event(scalar, vector, self.origin, self.frame)
        surface_event.insert_subfield(surface_key, vector)
        surface_event.insert_subfield(surface_key + '_lt', scalar)
        surface_event.insert_subfield('perp', vector.wod)

        if coords:
            surface_event.insert_subfield('coord1', scalar)
            surface_event.insert_subfield('coord2', scalar)
            surface_event.insert_subfield('coord3', scalar)

        return (surface_event, new_link)

    ############################################################################
    # Photon Solver based on coordinates at the surface
    ############################################################################

    def photon_to_event_by_coords(self, arrival, coords, derivs=False,
                                        guess=None, antimask=None,
                                        quick={}, converge={}):
        """Photon departure event from surface coordinates, given arrival event.

        See _solve_photon_by_coords() for details.
        """

        return self._solve_photon_by_coords(arrival, coords, -1, derivs,
                                            guess, antimask, quick, converge)

    #===========================================================================
    def photon_from_event_by_coords(self, departure, coords, derivs=False,
                                          guess=None, antimask=None,
                                          quick={}, converge={}):
        """Photon arrival event at surface coordinates, given departure event.

        See _solve_photon_by_coords() for details.
        """

        return self._solve_photon_by_coords(departure, coords, +1, derivs,
                                            guess, antimask, quick, converge)

    #===========================================================================
    def _solve_photon_by_coords(self, link, coords, sign, derivs=False,
                                      guess=None, antimask=None,
                                      quick={}, converge={}):
        """Solve for a photon surface intercept from event and coordinates.

        Input:
            link        the link event of a photon's arrival or departure.

            coords      a tuple of two or three coordinate values defining
                        locations at or near the surface.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the surface and arriving later at the
                           link.
                        +1 to return later events, corresponding to photons
                           departing from the link and arriving later at the
                           surface.

            derivs      True to propagate derivatives of the link position and
                        coordinates into the returned event. Derivatives with
                        respect to time are always retained.

            guess       an initial guess to use as the event time for the
                        surface; otherwise None. Should be used if the event
                        time was already returned from a similar calculation.

            antimask    if not None, this is a boolean array to be applied to
                        event times and positions. Only the indices where
                        antimask=True will be used in the solution.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         a tuple of two Events (surface_event, link_event).

            surface_event
                        the event on the surface that matches the light travel
                        time from the link event. This is event is defined in
                        the frame of the surface and relative to the surface's
                        origin.

            link_event  a copy of the given event, with the photon arrival or
                        departure line of sight and light travel time filled in.

            If sign is +1, then these subfields and derivatives are defined.
                In surface_event:
                    arr         direction of the arriving photon at the surface.
                    arr_lt      (negative) light travel time from the link
                                event to the surface.
                In link_event:
                    dep         direction of the departing photon at the event.
                    dep_lt      light travel time between the events.

            If sign is -1, then 'arr' and 'dep' are swapped for the two events.
            Note that subfield 'arr_lt' is always negative and 'dep_lt' is
            always positive. Subfields 'arr' and 'dep' have the same direction
            in both events.

        Convergence parameters are as follows:
            max_iterations  the maximum number of iterations of Newton's method
                            to perform. It should almost never need to be > 6.
            dlt_precision   iteration stops when the largest change in light
                            travel time between one iteration and the next falls
                            below this threshold (in seconds).
            dlt_limit       the maximum allowed absolute value of the change in
                            light travel time from the nominal range calculated
                            initially. Changes in light travel with absolute
                            values larger than this limit are clipped. This
                            prevents the divergence of the solution in some
                            cases.
        """

        # Handle derivatives
        if not derivs:
            link = link.wod         # preserves time-dependence
        # From here on, derivs=True in all calculations

        # Assemble convergence parameters
        if converge:
            defaults = SURFACE_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = SURFACE_PHOTONS.__dict__

        iters = converge['max_iterations']
        precision = converge['dlt_precision']
        limit = converge['dlt_limit']

        # Interpret the quick parameters
        if isinstance(quick, dict):
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Interpret the sign
        if sign < 0.:
            signed_c = -C
            surface_key = 'dep'
            link_key = 'arr'
        else:
            signed_c = C
            link_key = 'dep'
            surface_key = 'arr'

        # Define the antimask
        if antimask is None:
            antimask = link.antimask
        else:
            antimask = antimask & link.antimask

        # If the link is entirely masked...
        if not np.any(antimask):
            return self._fully_masked_result(link, link_key)

        # Shrink the event
        unshrunk_link = link
        link = link.shrink(antimask)

        # Define quantities with respect to SSB in J2000
        link_wrt_ssb = link.wrt_ssb(derivs=True, quick=quick)
        path_wrt_ssb = self.origin.wrt(Path.SSB, Frame.J2000)
        frame_wrt_j2000 = self.frame.wrt(Frame.J2000)

        # Prepare for iteration, avoiding any derivatives for now
        obs_wrt_ssb_now = link_wrt_ssb.state

        # Validate the guess input
        if guess is not None:
            guess = Scalar.as_scalar(guess, recursive=False).wod
            guess = guess.shrink(antimask)

            # Masked values in the guess are not usable
            if np.all(guess.mask):
                guess = None
            elif np.any(guess.mask):
                guess = guess.copy()
                guess[guess.mask] = guess.mean()

        # Prepare the first guesses at the surface_time and lt
        if guess is None:
            # If no guess was provided, base the time on the range to the origin
            link_time = link.time.wod
            origin_event = path_wrt_ssb.event_at_time(link_time, quick=quick)
            lt = (origin_event.pos.wod - obs_wrt_ssb_now.wod).norm() / signed_c
            surface_time = link_time + lt
        else:
            surface_time = guess
            lt = surface_time - link.time.wod

        # Set light travel time limits to avoid a diverging solution
        lt_min = lt.min(builtins=True) - limit
        lt_max = lt.max(builtins=True) + limit

        # For a non-virtual surface, pos_wrt_origin is fixed
        if not self.IS_VIRTUAL:
            pos_wrt_origin_frame = self.vector3_from_coords(coords,
                                                            time=surface_time,
                                                            derivs=True)

        # Iterate to solve for lt. Convergence is rapid because all speeds are
        # non-relativistic.
        max_dlt = np.inf
        converged = False
        for count in range(iters+1):

            # Quicken the path and frame as soon as the range of surface times
            # indicates that this would be beneficial.
            path_wrt_ssb = path_wrt_ssb.quick_path(surface_time, quick=quick)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quick)
                # Below, we specify quick=False because the path and frame are
                # already quickened.


            # Quicken the path and frame evaluations on first iteration
            # Below, we specify quick=False because it's already quick.
            path_wrt_ssb = path_wrt_ssb.quick_path(surface_time, quick=quick)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quick)

            # Evaluate the observer position relative to the current surface
            origin_wrt_ssb_then = path_wrt_ssb.event_at_time(surface_time,
                                                             quick=False).state
            obs_wrt_origin_j2000 = obs_wrt_ssb_now - origin_wrt_ssb_then

            # Locate the coordinate position relative to the current surface
            surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                              quick=False)
            if self.IS_VIRTUAL:
               obs_wrt_origin_frame = surface_xform.rotate(obs_wrt_origin_j2000,
                                                           derivs=True)
               pos_wrt_origin_frame = self.vector3_from_coords(coords,
                                                    obs=obs_wrt_origin_frame,
                                                    time=surface_time,
                                                    derivs=True)

            # Locate the coordinate position in J2000
            pos_wrt_origin_j2000 = surface_xform.unrotate(pos_wrt_origin_frame,
                                                          derivs=True)

            # Update the light travel time
            los_in_j2000 = pos_wrt_origin_j2000 - obs_wrt_origin_j2000
            new_lt = los_in_j2000.norm() / signed_c
            new_lt = new_lt.clip(lt_min, lt_max, False)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Surface.DEBUG:
                LOGGING.convergence('Surface._solve_photon_by_coords',
                                    'iter=%d; change=%.6g' % (count+1, max_dlt))

            if max_dlt <= precision:
                converged = True
                break

            if max_dlt >= prev_max_dlt:
                break

            # Re-evaluate the surface time
            surface_time = link.time + lt

        #### END OF LOOP

        if not converged:
            LOGGING.warn('Surface._solve_photon_by_coords did not converge;',
                         'iter=%d; change=%.6g' % (count+1, max_dlt))

        # Update the mask on light time to hide intercepts outside the defined
        # limits
        new_mask = ((lt.values * sign < 0.)
                    | (lt.values == lt_min)
                    | (lt.values == lt_max))
        if np.any(new_mask):
            lt = lt.remask_or(new_mask)

        surface_time = link.time + lt

        # If the link is entirely masked, return masked results
        if max_dlt < 0. or np.all(surface_time.mask):
            return self._fully_masked_result(unshrunk_link, link_key)

        # Determine the line of sight vector in J2000
        if sign < 0:
            los_in_j2000 = -los_in_j2000

        # Create the surface event in its own frame
        surface_event = Event(surface_time, pos_wrt_origin_frame,
                              self.origin, self.frame)

        los_in_frame = surface_xform.rotate(los_in_j2000)
        surface_event.insert_subfield(surface_key, los_in_frame)
        surface_event.insert_subfield(surface_key + '_lt', -lt)

        perp = self.normal(pos_wrt_origin_frame, time=surface_time, derivs=True)
        vflat = self.velocity(pos_wrt_origin_frame, time=surface_time)
        surface_event.insert_subfield('perp', perp)
        surface_event.insert_subfield('vflat', vflat)

        # Construct the updated link_event
        new_link = link.replace(link_key + '_j2000', los_in_j2000,
                                link_key + '_lt', lt)

        # Unshrink
        surface_event = surface_event.unshrink(antimask, unshrunk_link.shape)
        new_link = new_link.unshrink(antimask, unshrunk_link.shape)

        return (surface_event, new_link)

    ############################################################################
    # Photon Solver based on surface normal and remote event
    ############################################################################

    def photon_from_normal_to_event(self, arrival, derivs=False, guess=None,
                                          antimask=None, quick={}, converge={}):
        """Photon departure from this surface, given the arrival event and the
        requirement that it left along the surface normal.

        This can be used to solve for the sub-observer normal point on a
        surface. See _solve_normal_for_photon_event() for details of inputs.
        """

        return self._solve_normal_for_photon_event(arrival, -1, derivs, guess,
                                                   antimask, quick, converge)

    #===========================================================================
    def photon_from_event_to_normal(self, departure, derivs=False, guess=None,
                                          antimask=None, quick={}, converge={}):
        """Photon arrival at this surface, given the departure event and the
        requirement that it arrived along the surface normal.

        See _solve_normal_for_photon_event() for details.
        """

        return self._solve_normal_for_photon_event(departure, +1, derivs, guess,
                                                   antimask, quick, converge)

    #===========================================================================
    def _solve_normal_for_photon_event(self, link, sign, derivs=False,
                                             guess=None, antimask=None,
                                             quick={}, converge={}):
        """Solve for a the surface intercept event based on remote photon event
        and the requirement that the apparent photon path be normal to the
        surface.

        Input:
            link        the link event of a photon's arrival or departure.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the surface and arriving later at the
                           link.
                        +1 to return later events, corresponding to photons
                           departing from the link and arriving later at the
                           surface.

            derivs      True to propagate derivatives of the link position and
                        and line of sight into the returned event. Derivatives
                        with respect to time are always retained.

            guess       an initial guess to use as the event time for the
                        surface; otherwise None. Should only be used if the event
                        time was already returned from a similar calculation.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         a tuple (surface_event, link_event).

            surface_event
                        the event on the surface that matches the light travel
                        time from the link event. This is event is defined in
                        the frame of the surface and relative to the surface's
                        origin.

                        The surface event also contains three Scalar subfields,
                        "coord1", "coord2", and "coord3", containing the surface
                        coordinates at the intercept point (and their optional
                        derivatives).

            link_event  a copy of the given event, with the photon vector and
                        travel time filled in.

            If sign is +1, then these subfields and derivatives are defined.
                In path_event:
                    arr         direction of the arriving photon at the surface.
                    arr_ap      apparent direction of the arriving photon.
                    arr_lt      (negative) light travel time from the link
                                event to the surface.
                In link_event:
                    dep         departing photon direction to the surface.
                    dep_ap      apparent direction of the departing photon.
                    dep_lt      light travel time between the events.

            If sign is -1, then 'arr' and 'dep' are swapped for the two events.
            Note that subfield 'arr_lt' is always negative and 'dep_lt' is
            always positive. Subfields 'arr' and 'dep' have the same direction
            in both events.

            The subfields coord1, coord2, and coord3 are always defined in the
            surface event. These provide the coordinates of the surface
            intercept point.

        Convergence parameters are as follows:
            max_iterations  the maximum number of iterations of Newton's method
                            to perform. It should almost never need to be > 6.
            dlt_precision   iteration stops when the largest change in light
                            travel time between one iteration and the next falls
                            below this threshold (in seconds).
            dlt_limit       the maximum allowed absolute value of the change in
                            light travel time from the nominal range calculated
                            initially. Changes in light travel with absolute
                            values larger than this limit are clipped. This
                            prevents the divergence of the solution in some
                            cases.
        """

        #### TODO: full testing!!

        if self.IS_VIRTUAL:
            raise ValueError('Surface._solve_normal_for_photon_event ' +
                             ' does not support virtual surface class '
                             + type(self).__name__)

        # Handle derivatives
        if not derivs:
            link = link.wod     # preserves time-derivatives; removes others
        # From here on, derivs=True in all calculations

        # Assemble convergence parameters
        if converge:
            defaults = SURFACE_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = SURFACE_PHOTONS.__dict__

        iters = converge['max_iterations']
        precision = converge['dlt_precision']
        limit = converge['dlt_limit']

        # Interpret the quick parameters
        if isinstance(quick, dict):
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Interpret the sign
        if sign < 0.:
            signed_c = -C
            surface_key = 'dep'
            link_key = 'arr'
        else:
            signed_c = C
            link_key = 'dep'
            surface_key = 'arr'

        # Define the antimask
        if antimask is None:
            antimask = link.antimask
        else:
            antimask = antimask & link.antimask

        # If the link is entirely masked...
        if not np.any(antimask):
            return self._fully_masked_result(link, link_key)

        # Shrink the event
        unshrunk_link = link
        link = link.shrink(antimask)

        # Define the link event relative to the SSB in J2000
        link_wrt_ssb = link.wrt_ssb(derivs=True, quick=quick)

        obs_wrt_ssb_now = link_wrt_ssb.state
        los_in_j2000 = link_wrt_ssb.get_subfield(link_key)
        los_in_j2000 = los_in_j2000.with_norm(C)    # scale factor is lt

        # Validate the guess input
        if guess is not None:
            guess = Scalar.as_scalar(guess, recursive=False).wod
            guess = guess.shrink(antimask)

            # Masked values in the guess are not usable
            if np.all(guess.mask):
                guess = None
            elif np.any(guess.mask):
                guess = guess.copy()
                guess[guess.mask] = guess.mean()

        # Make an initial guess at the light travel time
        origin_wrt_ssb = self.origin.wrt(Path.SSB, Frame.J2000)
        if guess is None:
            # If no guess was provided, base the time on the range to the origin
            link_time = link.time.wod
            origin_event = origin_wrt_ssb.event_at_time(link_time, quick=quick)
            lt = (origin_event.pos.wod - link_wrt_ssb.pos.wod).norm() / signed_c
            surface_time = link_time + lt
        else:
            surface_time = guess
            lt = surface_time - link.time.wod

        # Define the surface path and frame relative to the SSB in J2000, quicken
        origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time, quick=quick)
        frame_wrt_j2000 = self.frame.wrt(Frame.J2000)
        frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                      quick=quick)

        # Set light travel time limits to avoid a diverging solution
        lt_min = lt.min(builtins=True) - limit
        lt_max = lt.max(builtins=True) + limit

        # Iterate to solve for lt. Convergence is rapid because all speeds are
        # non-relativistic
        max_dlt = np.inf
        converged = False
        hints = True                        # Speeds up some calculations
        for count in range(iters):

            # Evaluate the observer position relative to the current surface
            origin_wrt_ssb_then = origin_wrt_ssb.event_at_time(
                                                            surface_time).state
            obs_wrt_origin_j2000 = obs_wrt_ssb_now - origin_wrt_ssb_then

            # Locate the coordinate position relative to the current surface
            surface_xform = frame_wrt_j2000.transform_at_time(surface_time)
            obs_wrt_origin_frame = surface_xform.rotate(obs_wrt_origin_j2000,
                                                        derivs=True)

            # Update the intercept times; save the intercept normal positions
            (cept_in_frame,
             new_lt, hints) = self.intercept_normal_to(obs_wrt_origin_frame,
                                                       time=surface_time,
                                                       direction=surface_key,
                                                       derivs=True,
                                                       guess=lt,
                                                       hints=hints)

            new_lt = new_lt.clip(lt_min, lt_max, remask=False)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Surface.DEBUG:
                LOGGING.convergence('Surface._solve_normal_for_photon_event',
                                    'iter=%d; change=%.6g' % (count+1, max_dlt))

            if max_dlt <= precision:
                converged = True
                break

            if max_dlt >= prev_max_dlt:
                break

            # Re-evaluate the surface time
            surface_time = link.time + lt

        #### END OF LOOP

        if not converged:
            LOGGING.warn('Surface._solve_normal_for_photon_event ' +
                         'did not converge;',
                         'iter=%d; change=%.6g' % (count+1, max_dlt))

        # Update the mask on light time to hide intercepts outside the defined
        # limits
        new_mask = ((lt.values * sign < 0.)
                    | (lt.values == lt_min)
                    | (lt.values == lt_max))
        if np.any(new_mask):
            lt = lt.remask_or(new_mask)

        surface_time = link.time + lt

        # If the link is entirely masked, return masked results
        if max_dlt < 0. or np.all(surface_time.mask):
            return self._fully_masked_result(unshrunk_link, link_key,
                                             coords=True)

        #### Create the surface event in its own frame

        # The intercept event with respect to the surface has a time-derivative
        # due to the rate of change of the observer position. However, THIS IS
        # NOT A PHYSICAL VELOCITY. To define the surface event properly, we need
        # to remove the time derivative of cept_in_frame. We assign it a new
        # name d_dT to distinguish it from d_dt.

        event_state = cept_in_frame.rename_deriv('t', 'T', method='add')
        event_time  = surface_time.rename_deriv('t', 'T', method='add')
        surface_event = Event(event_time, event_state, self.origin, self.frame)

        # Fill in standard subfields

        # To calculate the time-dependence of other attributes, we need to use
        # the original cept_in_frame in order to give them the correct time-
        # dependence. This is OK because these are not understood to be physical
        # velocities.

        alt_event = Event(surface_time, cept_in_frame,
                          self.origin, self.frame)
        los_in_j2000 = sign * (alt_event.ssb.state - obs_wrt_ssb_now)
        surface_event.insert_subfield(surface_key + '_j2000', los_in_j2000)
        surface_event.insert_subfield(surface_key + '_lt', -lt)

        perp = self.normal(cept_in_frame, time=surface_time, derivs=True)
        vflat = self.velocity(cept_in_frame, surface_time)
        surface_event.insert_subfield('perp', perp)
        surface_event.insert_subfield('vflat', vflat)

        # Fill in coordinate subfields
        coords = self.coords_from_vector3(cept_in_frame, obs_wrt_origin_frame,
                                          time=surface_time, axes=3,
                                          derivs=True, hints=hints)
        surface_event.insert_subfield('coord1', coords[0])
        surface_event.insert_subfield('coord2', coords[1])
        surface_event.insert_subfield('coord3', coords[2])

        # Save the hints if any
        if hints is not True:
            surface_event.insert_subfield('hints', hints)

        # Construct the updated link_event
        new_link = link.replace(link_key + '_j2000', los_in_j2000,
                                link_key + '_lt', lt)

        # Unshrink
        surface_event = surface_event.unshrink(antimask, unshrunk_link.shape)
        new_link = new_link.unshrink(antimask, unshrunk_link.shape)

        return (surface_event, new_link)

    ############################################################################
    # Photon Solver based on surface normal event and remote path
    ############################################################################

    def photon_from_path_to_normal(self, time, path, derivs=False, guess=None,
                                         antimask=None, quick={}, converge={}):
        """Photon departure event from a path given the requirement that it
        arrive at the surface at the specified time along a surface normal.

        This can be used to solve for the sub-solar point on a surface. See
        _solve_photon_normal_to_surface() for details of inputs.
        """

        return self._solve_photon_normal_to_surface(time, path, -1, derivs,
                                                    guess, antimask, quick,
                                                    converge)

    #===========================================================================
    def photon_from_normal_to_path(self, time, path, derivs=False, guess=None,
                                         antimask=None, quick={}, converge={}):
        """Photon arrival at this surface, given departure and surface normal
        requirement.

        See _solve_photon_normal_to_surface() for details.
        """

        return self._solve_photon_normal_to_surface(time, path, +1, derivs,
                                                    guess, antimask, quick,
                                                    converge)

    #===========================================================================
    def _solve_photon_normal_to_surface(self, time, path, sign, derivs=False,
                                              guess=None, antimask=None,
                                              quick={}, converge={}):
        """Solve for a photon surface intercept based on remote path and local
        surface normal.

        Input:
            time        time at the surface for the photon event.

            path        remote path for the event associated with the photon's
                        travel.

            sign        -1 to return earlier path events, corresponding to
                           photons departing from the path and arriving later at
                           the surface.
                        +1 to return later path events, corresponding to photons
                           departing from the surface and arriving later at the
                           path.

            derivs      True to propagate derivatives of the link position and
                        and line of sight into the returned event. Derivatives
                        with respect to time are always retained.

            guess       an initial guess to use as the event time for the
                        path; otherwise None. Should only be used if the event
                        time was already returned from a similar calculation.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         a tuple (surface_event, link_event).

            surface_event
                        the event on the surface that matches the light travel
                        time from the linked path. This is event is defined in
                        the frame of the surface and relative to the surface's
                        origin.

                        The surface event also contains three Scalar subfields,
                        "coord1", "coord2", and "coord3", containing the surface
                        coordinates at the intercept point (and their optional
                        derivatives).

            path_event  the event at the remote path.

            If sign is +1, then these subfields and derivatives are defined.
                In surface_event:
                    dep         departing photon direction at the surface.
                    dep_ap      apparent direction of the departing photon.
                    dep_lt      light travel time between the events.

                In path_event:
                    arr         direction of the arriving photon from the
                                surface.
                    arr_ap      apparent direction of the arriving photon.
                    arr_lt      (negative) light travel time between the events.

            If sign is -1, then 'arr' and 'dep' are swapped for the two events.
            Note that subfield 'arr_lt' is always negative and 'dep_lt' is
            always positive. Subfields 'arr' and 'dep' have the same direction
            in both events.

        Convergence parameters are as follows:
            max_iterations  the maximum number of iterations of Newton's method
                            to perform. It should almost never need to be > 6.
            dlt_precision   iteration stops when the largest change in light
                            travel time between one iteration and the next falls
                            below this threshold (in seconds).
            dlt_limit       the maximum allowed absolute value of the change in
                            light travel time from the nominal range calculated
                            initially. Changes in light travel with absolute
                            values larger than this limit are clipped. This
                            prevents the divergence of the solution in some
                            cases.
        """

        #### TODO: full testing!!

        if self.IS_VIRTUAL:
            raise ValueError('Surface._solve_photon_normal_to_surface ' +
                             ' does not support virtual surface class '
                             + type(self).__name__)

        # Handle derivatives
        if not derivs:
            time = time.without_derivs(preserve='t')
        # From here on, derivs=True in all calculations

        # Assemble convergence parameters
        if converge:
            defaults = SURFACE_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = SURFACE_PHOTONS.__dict__

        iters = converge['max_iterations']
        precision = converge['dlt_precision']
        limit = converge['dlt_limit']

        # Interpret the quick parameters
        if isinstance(quick, dict):
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Interpret the sign
        if sign < 0.:
            surface_key = 'arr'
            remote_key = 'dep'
        else:
            remote_key = 'arr'
            surface_key = 'dep'

        # Define the antimask
        if antimask is None:
            antimask = time.antimask
        else:
            antimask = antimask & time.antimask

        # Create a placeholder event for error situations
        unshrunk_remote = Event(time, Vector3.ZERO, path, path.frame)

        if not np.any(antimask):        # entirely masked input
            return self._fully_masked_result(unshrunk_remote, remote_key)

        # Shrink the time
        surface_time = time.shrink(antimask)

        # Validate the guess input
        if guess is not None:
            guess = Scalar.as_scalar(guess, recursive=False).wod
            guess = guess.shrink(antimask)

            # Masked values in the guess are not usable
            if np.all(guess.mask):
                guess = None
            elif np.any(guess.mask):
                guess = guess.copy()
                guess[guess.mask] = guess.mean()

        # Make an initial guess at the light travel time
        if guess is None:
            # If no guess was provided, base the time on the separation distance
            origin_event = Event(surface_time.wod, Vector3.ZERO,
                                 self.path, self.frame)
            (path_event, _) = path._solve_photon(origin_event, -sign,
                                                 quick=quick, converge=converge)
            path_time = path_event.time.wod
        else:
            path_time = guess

        lt = path_time - surface_time.wod

        # Lock down the surface origin and frame relative to SSB/J2000
        surface_xform = self.frame.transform_at_time(surface_time)
        origin_wrt_ssb = self.path.event_at_time(surface_time).state

        # Define the path relative to SSB/J2000 and quicken
        path_wrt_ssb = path.wrt(Path.SSB, Frame.J2000)
        path_wrt_ssb = path_wrt_ssb.quick_path(path_time, quick=quick)

        # Iterate to solve for lt. Convergence is rapid because all speeds are
        # non-relativistic.
        max_dlt = np.inf
        converged = False
        hints = True                        # Speeds up some calculations
        for count in range(iters):

            # Locate position relative to origin in SSB/J2000
            pos_wrt_origin_j2000 = (path_wrt_ssb.event_at_time(path_time).state
                                    - origin_wrt_ssb)

            # Locate position relative to origin in surface frame
            pos_wrt_origin_frame = surface_xform.rotate(pos_wrt_origin_j2000,
                                                        derivs=True)

            # Update the intercepts
            (cept_in_frame,
             new_lt, hints) = self.intercept_normal_to(pos_wrt_origin_frame,
                                                       time=surface_time,
                                                       direction=surface_key,
                                                       derivs=True,
                                                       guess=lt,
                                                       hints=hints)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Surface.DEBUG:
                LOGGING.convergence('Surface._solve_photon_normal_to_surface',
                                    'iter=%d; change=%.6g' % (count+1, max_dlt))

            if max_dlt <= precision:
                converged = True
                break

            if max_dlt >= prev_max_dlt:
                break

            # Re-evaluate the path time
            path_time = surface_time + lt

        #### END OF LOOP

        if not converged:
            LOGGING.warn('Surface._solve_photon_normal_to_surface ' +
                         'did not converge;',
                         'iter=%d; change=%.6g' % (count+1, max_dlt))

        # If the result is entirely masked, return masked results
        if max_dlt < 0. or np.all(path_time.mask):
            # This is a fake, fully masked link
            vec = Vector3.ZERO.broadcast_to(path_time.shape)
            link = Event(path_time.remask(True), (vec, vec),
                         path=path, frame=Frame.J2000)
            return self._fully_masked_result(link, remote_key, coords=False)

        #### Create the surface event in its own frame

        # The intercept event with respect to the surface has a time-derivative
        # due to the rate of change of the observer position. However, THIS IS
        # NOT A PHYSICAL VELOCITY. To define the surface event properly, we need
        # to remove the time derivative of cept_in_frame. We assign it a new
        # name d_dT to distinguish it from d_dt.

        event_state = cept_in_frame.rename_deriv('t', 'T', method='add')
        event_time  = surface_time.rename_deriv('t', 'T', method='add')
        surface_event = Event(event_time, event_state, self.path, self.frame)

        # Subfields are calculated using the original cept_in_frame, so these
        # attributes will have correct time-derivatives. This is OK because
        # these time-derivatives are not physical velocities.

        normal = self.normal(cept_in_frame, time=surface_time, derivs=True)
        surface_event.insert_subfield(surface_key + '_ap', normal)
        surface_event.insert_subfield(surface_key + '_lt', -lt)
        surface_event.insert_subfield('perp', normal)
        surface_event.insert_subfield('vflat', self.velocity(cept_in_frame,
                                                             surface_time))

        # Fill in coordinate subfields
        coords = self.coords_from_vector3(cept_in_frame, cept_in_frame,
                                          time=surface_time, axes=3,
                                          derivs=True, hints=hints)
        surface_event.insert_subfield('coord1', coords[0])
        surface_event.insert_subfield('coord2', coords[1])
        surface_event.insert_subfield('coord3', coords[2])

        # Save the hints if any
        if hints is not True:
            surface_event.insert_subfield('hints', hints)

        # Create the remote event
        remote_event = path.event_at_time(path_time)

        los_in_j2000 = surface_event.get_subfield(surface_key + '_j2000')
        remote_event.insert_subfield(remote_key + '_j2000', los_in_j2000)
        remote_event.insert_subfield(remote_key + '_lt', lt)

        return (surface_event, remote_event)

    ############################################################################
    # Class Method
    ############################################################################

    @staticmethod
    def resolution(dpos_duv, _unittest=False):
        """Determine the spatial resolution on a surface.

        Input:
            dpos_duv    A Vector3 with denominator shape (2,), defining the
                        partial derivatives d(x,y,z)/d(u,v), where (x,y,z) are
                        the 3-D coordinates of a point on the surface, and (u,v)
                        are pixel coordinates.

        Return:         A tuple (res_min, res_max) where:
            res_min     A Scalar containing resolution values (km/pixel) in the
                        direction of finest spatial resolution.
            res_max     A Scalar containing resolution values (km/pixel) in the
                        direction of coarsest spatial resolution.

        Note: For the best solution, the derivatives should be adjusted such
        that the u-axis and the v-axis are locally perpendicular. See the source
        code of Backplane.dlos_duv1 in backplane/__init__.py for details.
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

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Surface(unittest.TestCase):

    def runTest(self):

        np.random.seed(6631)

        # Most methods are heavily tested elsewhere

        # Surface.resolution...

        # Make sure the rotated resolution vectors are perpendicular
        dpos_duv = Vector3(np.random.randn(10000, 3, 2), drank=1)
        (new_du, new_dv) = Surface.resolution(dpos_duv, _unittest=True)
        self.assertTrue(new_du.dot(new_dv).max() < 1.e-12)

        # We also expect area to be conserved
        dpos_du = Vector3(dpos_duv.values[...,0])
        dpos_dv = Vector3(dpos_duv.values[...,1])
        diffs = dpos_du.cross(dpos_dv) - new_du.cross(new_dv)
        self.assertTrue(diffs.norm().max() < 1.e-14)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
