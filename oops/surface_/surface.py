################################################################################
# oops/surface_/surface.py: Abstract class Surface
################################################################################

import numpy as np
from polymath import *

from oops.config     import QUICK, SURFACE_PHOTONS, LOGGING
from oops.event      import Event
from oops.path_.path import *
import oops.constants as constants

class Surface(object):
    """Surface is an abstract class describing a 2-D object that moves and
    rotates in space. A surface employs an internal coordinate system, not
    necessarily rectangular, in which two primary coordinates define locations
    on the surface, and an optional third coordinate can define points above or
    below that surface. The shape is always fixed.

    Required attributes:
        origin      the waypoint of the path defining the surface's center.
        frame       the wayframe of the frame in which the surface is defined.
    """

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = True
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = True
    intercept_DERIVS_ARE_IMPLEMENTED = True
    normal_DERIVS_ARE_IMPLEMENTED = True
    intercept_with_normal_DERIVS_ARE_IMPLEMENTED = True
    intercept_normal_to_DERIVS_ARE_IMPLEMENTED = True

    ########################################
    # Each subclass must override...
    ########################################

    def __init__(self):
        """Constructor for a Surface object.
        """

        pass

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

        pass

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        pass

    def intercept(self, obs, los, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        pass

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        pass

    ########################################
    # Optional Methods...
    ########################################

    def intercept_with_normal(self, normal, derivs=False, guess=None):
        """Intercept point where the normal vector parallels the given vector.

        Input:
            normal      a Vector3 of normal vectors.
            derivs      True to propagate derivatives in the normal vector into
                        the returned intercepts.
            guess       optional initial guess a coefficient array p such that:
                            pos = intercept + p * normal(intercept);
                        use guess=False for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         a Vector3 of surface intercept points, in km. Where no
                        solution exists, the returned Vector3 will be masked.

                        If guess is not None, then it instead returns a tuple
                        (intercepts, p), where p is the converged solution such
                        that 
                            pos = intercept + p * normal(intercept).
        """

        raise NotImplementedError("intercept_with_normal() not implemented " +
                                  "for class " + type(self).__name__)

    def intercept_normal_to(self, pos, derivs=False, guess=None):
        """Intercept point whose normal vector passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface.
            derivs      True to propagate derivatives in pos into the returned
                        intercepts.
            guess       optional initial guess a coefficient array p such that:
                            intercept = pos + p * normal(intercept);
                        use guess=False for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         a vector3 of surface intercept points, in km. Where no
                        solution exists, the returned vector will be masked.

                        If guess is not None, then it instead returns a tuple
                        (intercepts, p), where p is the converged solution such
                        that 
                            intercept = pos + p * normal(intercept).
        """

        raise NotImplementedError("intercept_normal_to() not implemented " +
                                  "for class " + type(self).__name__)

    def velocity(self, pos):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface.

        Return:         a Vector3 of velocities, in units of km/s.
        """

        return Vector3.ZERO

    ############################################################################
    # Photon Solver based on line of sight
    ############################################################################

    def photon_to_event(self, arrival, derivs=False, guess=None,
                              quick={}, converge={}):
        """Photon departure from this surface, given arrival and line of sight.

        See _solve_photon_by_los() for details."""

        return self._solve_photon_by_los(arrival, -1, derivs, guess,
                                                      quick, converge)

    def photon_from_event(self, departure, derivs=False, guess=None,
                                quick={}, converge={}):
        """Photon arrival at this surface, given departure and line of sight.

        See _solve_photon_by_los() for details."""

        return self._solve_photon_by_los(departure, +1, derivs, guess,
                                                        quick, converge)

    def _solve_photon_by_los(self, link, sign, derivs=False, guess=None,
                                               quick={}, converge={}):
        """Solve for a photon surface intercept from event and line of sight.

        Input:
            link        the link event of a photon's arrival or departure.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the surface and arriving later at the
                           link.
                        +1 to return later events, corresponding to photons
                           departing from the link and arriving later at the
                           surface.

            derivs      True to propagate derivatives of the link time, position
                        and line of sight into the returned event.

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

            link_event  a copy of the given event, with the photon travel time
                        filled in.

            If sign is +1, then these subfields and derivatives are defined.
                In path_event:
                    arr         direction of the arriving photon at the surface.
                    arr_ap      apparent direction of the arriving photon.
                    arr_lt      (negative) light travel time from the link
                                event to the surface.
                In link_event:
                    dep_lt      light travel time between the events.

            If sign is -1, then 'arr' and 'dep' are swapped for the two events.
            Note that subfield 'arr_lt' is always negative and 'dep_lt' is
            always positive. Subfields 'arr' and 'dep' have the same direction
            in both events.

        Convergence parameters are as follows:
            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. This prevents the
                        divergence of the solution in some cases.
        """

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
        collapse_threshold = converge['collapse_threshold']

        # Interpret the quick parameters
        if type(quick) == dict:
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Interpret the sign
        signed_c = sign * constants.C
        if sign < 0.:
            surface_key = 'dep'
            link_key = 'arr'
        else:
            link_key = 'dep'
            surface_key = 'arr'

        # If the link is entirely masked...
        if np.all(link.time.mask):
            surface_event = link.all_masked(origin=self.origin,
                                            frame=self.frame.wayframe,
                                            derivs=derivs)
            surface_event.insert_subfield(surface_key, Vector3.MASKED)
            surface_event.insert_subfield(surface_key + '_lt', Scalar.MASKED)

            new_link = link.replace(link_key + '_lt', Scalar.MASKED)

            return (surface_event, new_link)

        # Define the link event relative to the SSB in J2000
        link_wod = link.without_derivs()
        link_time = link_wod.time
        link_wrt_ssb = link.wrt_ssb(derivs=derivs, quick=quick)
        obs_wrt_ssb = link_wrt_ssb.pos
        los_wrt_ssb = link_wrt_ssb.get_subfield(link_key).without_derivs()
        los_wrt_ssb = los_wrt_ssb.unit() * constants.C  # scale factor is lt

        # Define the surface path and frame relative to the SSB in J2000
        origin_wrt_ssb  = self.origin.wrt(Path.SSB, Frame.J2000)
        frame_wrt_j2000 = self.frame.wrt(Frame.J2000)

        # Make an initial guess at the light travel time
        if guess is not None:
            surface_time = guess
            lt = surface_time - link_time
        else:
            # If no guess was provided, base the time on the range to the origin
            lt = (origin_wrt_ssb.event_at_time(link_time, quick=quick).pos -
                  obs_wrt_ssb).norm() / signed_c
            surface_time = link_time + lt

        # Set light travel time limits to avoid a diverging solution
        lt_min = (surface_time - link_time).min() - limit
        lt_max = (surface_time - link_time).max() + limit

        # Iterate. Convergence is rapid because all speeds are non-relativistic
        max_dlt = np.inf
        new_lt = None
        for iter in range(iters):

            # Quicken the path and frame evaluations (but no more than once)
            origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time,
                                                       quick=quick)

            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quick)

            # Locate the photons relative to the current origin in SSB/J2000
            pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb -
                            origin_wrt_ssb.event_at_time(surface_time,
                                                         quick=False).pos)

            # Rotate into the surface-fixed frame
            surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                              quick=False)
            pos_wrt_surface = surface_xform.rotate(pos_in_j2000, derivs=False)
            los_wrt_surface = surface_xform.rotate(los_wrt_ssb, derivs=False)
            obs_wrt_surface = pos_wrt_surface - lt * los_wrt_surface

            # Update the intercept times; save the intercept positions
            (pos_wrt_surface, new_lt) = self.intercept(obs_wrt_surface,
                                                       los_wrt_surface,
                                                       derivs=False,
                                                       guess=new_lt)

            new_lt = new_lt.clip(lt_min, lt_max, False)

            tmin = new_lt.min()
            tmax = new_lt.max()
            span = tmax - tmin

            collapsed_mask = (span == Scalar.MASKED)

            if span <= collapse_threshold:
                if LOGGING.surface_time_collapse:
                    print LOGGING.prefix, "Surface.collapse_time()",
                    print tmin, tmax - tmin
                new_lt = Scalar((tmin + tmax)/2., collapsed_mask, new_lt.units)

            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations:
                print LOGGING.prefix, "Surface._solve_photon_by_los",
                print iter, max_dlt

            if max_dlt <= precision or max_dlt >= prev_max_dlt or \
               max_dlt == Scalar.MASKED:
                    break

            # Re-evaluate the surface time
            surface_time = link_time + lt

        #### END OF LOOP

        # Update the mask on light time to hide intercepts outside the defined
        # limits
        mask = (link.mask | lt.mask | (lt.vals * sign < 0.) |
                (lt.vals == lt_min) | (lt.vals == lt_max))
        if not np.any(mask): mask = False

        # The remasking will fail if lt has been time-collapsed
        try:
            lt = lt.remask(mask)
        except ValueError:
            pass

        surface_time = link_time + lt

        # Put the derivatives back if necessary
        if derivs:
            obs_wrt_ssb = link_wrt_ssb.state
            los_wrt_ssb = link_wrt_ssb.get_subfield(link_key).unit() * \
                                                        constants.C
            pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb -
                            origin_wrt_ssb.event_at_time(surface_time,
                                                         quick=False).state)

        else:
            pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb -
                            origin_wrt_ssb.event_at_time(surface_time,
                                                         quick=False).pos)

            pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb -
                            origin_wrt_ssb.event_at_time(surface_time,
                                                         quick=False).pos)

        # Re-rotate into the surface-fixed frame
        surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                          quick=False)
        pos_wrt_surface = surface_xform.rotate(pos_in_j2000, derivs)
        los_wrt_surface = surface_xform.rotate(los_wrt_ssb, derivs)
        obs_wrt_surface = pos_wrt_surface - lt * los_wrt_surface

        # Update the intercept time and position, with derivatives if necessary
        (pos_wrt_surface, lt) = self.intercept(obs_wrt_surface,
                                               los_wrt_surface,
                                               derivs=derivs,
                                               guess=lt)

        # Update the mask on light time to hide intercepts behind the observer
        # or outside the defined limits
        mask = (link.mask | lt.mask | (lt.vals * sign < 0.) |
                (lt.vals == lt_min) | (lt.vals == lt_max))
        if not np.any(mask): mask = False

        # The remasking will fail if lt has been time-collapsed
        try:
            lt = lt.remask(mask)
        except ValueError:
            pass

        pos_wrt_surface = pos_wrt_surface.remask(mask)

        # Create the surface event in its own frame
        surface_event = Event(link.time + lt, (pos_wrt_surface,Vector3.ZERO),
                              self.origin, self.frame)
        surface_event.collapse_time()

        surface_event.insert_subfield('perp',
                                      self.normal(pos_wrt_surface, derivs))
        surface_event.insert_subfield('vflat', self.velocity(pos_wrt_surface))
        surface_event.insert_subfield(surface_key, los_wrt_surface)
        surface_event.insert_subfield(surface_key + '_lt', -lt)

        # Fill in coordinate subfields
        (coord1,
         coord2,
         coord3) = self.coords_from_vector3(surface_event.state,
                                            obs_wrt_surface,
                                            axes=3, derivs=derivs)

        surface_event.insert_subfield('coord1', coord1)
        surface_event.insert_subfield('coord2', coord2)
        surface_event.insert_subfield('coord3', coord3)

        # Constructed the updated link_event
        new_link = link.replace(link_key + '_lt', lt)

        return (surface_event, new_link)

    ############################################################################
    # Photon Solver based on coordinates at the surface
    ############################################################################

    def photon_to_event_by_coords(self, arrival, coords, derivs=False,
                                        guess=None, quick={}, converge={}):
        """Photon departure event from surface coordinates, given arrival event.

        See _solve_photon_by_coords() for details."""

        return self._solve_photon_by_coords(arrival, coords, -1, derivs,
                                            guess, quick, converge)

    def photon_from_event_by_coords(self, departure, coords, derivs=False,
                                          guess=None, quick={}, converge={}):
        """Photon arrival event at surface coordinates, given departure event.

        See _solve_photon_by_coords() for details."""

        return self._solve_photon_by_coords(departure, coords, +1, derivs,
                                            guess, quick, converge)

    def _solve_photon_by_coords(self, link, coords, sign, derivs=False,
                                      guess=None, quick={}, converge={}):
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

            derivs      True to propagate derivatives of the link time, position
                        and coordinates into the returned event.

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

        Return:         a tuple of two Events (surface_event, link_event).

            surface_event
                        the event on the surface that matches the light travel
                        time from the link event. This is event is defined in
                        the frame of the surface and relative to the surface's
                        origin.

            link_event  a copy of the given event, with the photon arrival or
                        departure line of sight and light travel time filled in.

            If sign is +1, then these subfields and derivatives are defined.
                In path_event:
                    arr         direction of the arriving photon at the surface.
                    arr_ap      apparent direction of the arriving photon.
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
            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. This prevents the
                        divergence of the solution in some cases.
        """

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
        if type(quick) == dict:
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Interpret the sign
        signed_c = sign * constants.C
        if sign < 0.:
            surface_key = 'dep'
            link_key = 'arr'
        else:
            link_key = 'dep'
            surface_key = 'arr'

        # If the link is entirely masked...
        if np.all(link.time.mask):
            surface_event = link.all_masked(origin=self.origin,
                                            frame=self.frame.wayframe,
                                            derivs=derivs)
            surface_event.insert_subfield(surface_key, Vector3.MASKED)
            surface_event.insert_subfield(surface_key + '_lt', Scalar.MASKED)

            new_link = link.replace(link_key, Vector3.MASKED,
                                    link_key + '_lt', Scalar.MASKED)

            return (surface_event, new_link)

        # Define the link event relative to the SSB in J2000
        link_wod = link.without_derivs()
        link_time = link_wod.time
        link_wrt_ssb = link.wrt_ssb(derivs=derivs, quick=quick)
        obs_wrt_ssb = link_wrt_ssb.pos

        # Define the surface path and frame relative to the SSB in J2000
        origin_wrt_ssb  = self.origin.wrt(Path.SSB, Frame.J2000)
        frame_wrt_j2000 = self.frame.wrt(Frame.J2000)

        # Define the observer in the SSB frame
        link_wrt_ssb = link.wrt_ssb(quick)
        obs_wrt_ssb_now = link_wrt_ssb.pos

        # Make an initial guess at the light travel time
        if guess is not None:
            surface_time = guess
            lt = surface_time - link_time
        else:
            # If no guess was provided, base the time on the range to the origin
            lt = (origin_wrt_ssb.event_at_time(link_time, quick=quick).pos -
                  obs_wrt_ssb).norm() / signed_c
            surface_time = link_time + lt

        # Set light travel time limits to avoid a diverging solution
        lt_min = (surface_time - link_time).min() - limit
        lt_max = (surface_time - link_time).max() + limit

        # For a non-virtual surface, pos_wrt_origin is fixed
        if not self.IS_VIRTUAL:
            pos_wrt_origin_frame = self.vector3_from_coords(coords)

        # Iterate. Convergence is rapid because all speeds are non-relativistic
        max_dlt = np.inf
        new_lt = None
        converged = False
        for iter in range(iters+1):

            # Quicken the path and frame evaluations (but no more than once)
            origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time,
                                                       quick=quick)

            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quick)

            # Evaluate the observer position relative to the current surface
            origin_wrt_ssb_then = origin_wrt_ssb.event_at_time(surface_time,
                                                               quick=False).pos
            obs_wrt_origin_j2000 = obs_wrt_ssb_now - origin_wrt_ssb_then

            # Locate the coordinate position relative to the current surface
            surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                              quick=False)

            if self.IS_VIRTUAL:
               obs_wrt_origin_frame = surface_xform.rotate(obs_wrt_origin_j2000)
               pos_wrt_origin_frame = self.vector3_from_coords(coords,
                                                           obs_wrt_origin_frame)

            # Locate the coordinate position in J2000
            pos_wrt_origin_j2000 = surface_xform.unrotate(pos_wrt_origin_frame)

            # Update the light travel time
            los_in_j2000 = (pos_wrt_origin_j2000 + origin_wrt_ssb_then -
                            obs_wrt_ssb_now)
            new_lt = los_in_j2000.norm() / signed_c

            if converged: break

            new_lt = new_lt.clip(lt_min, lt_max, False)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations:
                print LOGGING.prefix, "Surface._solve_photon_by_coords",
                print iter, max_dlt

            if max_dlt <= precision or max_dlt >= prev_max_dlt or \
               max_dlt == Scalar.MASKED:
                    converged = True        # one final pass after convergence

            # Re-evaluate the surface time
            surface_time = link_time + lt

        # Update the mask on light time to hide intercepts outside the defined
        # limits
        mask = (link.mask | lt.mask | (lt.vals * sign < 0.) |
                (lt.vals == lt_min) | (lt.vals == lt_max))
        if not np.any(mask): mask = False

        if lt.shape != (): lt = lt.remask(mask)
        surface_time = link_time + lt

        # Determine the line of sight vector in J2000
        if sign < 0:
            los_in_j2000 = -los_in_j2000

        # Create the surface event in its own frame
        surface_event = Event(surface_time, (pos_wrt_origin_frame,Vector3.ZERO),
                              self.origin, self.frame)
        surface_event.collapse_time()

        los_in_frame = surface_xform.rotate(los_in_j2000)
        surface_event.insert_subfield(surface_key, los_in_frame)
        surface_event.insert_subfield(surface_key + '_lt', -lt)

        surface_event.insert_subfield('perp',
                                      self.normal(pos_wrt_origin_frame))
        surface_event.insert_subfield('vflat',
                                      self.velocity(pos_wrt_origin_frame))

        # Fill in derivatives if necessary
        if derivs:
            raise NotImplementedException("Derivatives are not implemented " +
                                          "for _solve_photon_by_coords()")

        # Constructed the updated link_event
        new_link = link.replace(link_key + '_j2000', los_in_j2000,
                                link_key + '_lt', lt)

        return (surface_event, new_link)

    ############################################################################
    # Class Method
    ############################################################################

    @staticmethod
    def resolution(dpos_duv):
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
        """

        # Define vectors parallel to the surface, containing the derivatives
        # with respect to each pixel coordinate.
        dpos_du = Vector3(dpos_duv.values[...,0], dpos_duv.mask)
        dpos_dv = Vector3(dpos_duv.values[...,1], dpos_duv.mask)

        # The resolution should be independent of the rotation angle of the
        # grid. We therefore need to solve for an angle theta such that
        #   dpos_du' = cos(theta) dpos_du - sin(theta) dpos_dv
        #   dpos_dv' = sin(theta) dpos_du + cos(theta) dpos_dv
        # where
        #   dpos_du' <dot> dpos_dv' = 0
        #
        # Then, the magnitudes of dpos_du' and dpos_dv' will be the local values
        # of finest and coarsest spatial resolution.
        #
        # Laying aside the overall scaling for a moment, instead solve:
        #   dpos_du' =   dpos_du - t dpos_dv
        #   dpos_dv' = t dpos_du +   dpos_dv
        # for which the dot product is zero.
        #
        # 0 =   t^2 (dpos_du <dot> dpos_dv)
        #     + t   (|dpos_dv|^2 - |dpos_du|^2)
        #     -     (dpos_du <dot> dpos_dv)
        #
        # Use the quadratic formula.

        a = dpos_du.dot(dpos_dv)
        b = dpos_dv.dot(dpos_dv) - dpos_du.dot(dpos_du)
        # c = -a

        # discr = b**2 - 4*a*c
        discr = b**2 + 4*a**2
        t = (discr.sqrt() - b)/ (2*a)

        # Now normalize and construct the primed partials
        cos_theta = 1. / (1 + t**2).sqrt()
        sin_theta = t * cos_theta

        dpos_du_prime_norm = (cos_theta * dpos_du - sin_theta * dpos_dv).norm()
        dpos_dv_prime_norm = (sin_theta * dpos_du + cos_theta * dpos_dv).norm()

        minres = Scalar(np.minimum(dpos_du_prime_norm.values,
                                   dpos_dv_prime_norm.values), dpos_duv.mask)
        maxres = Scalar(np.maximum(dpos_du_prime_norm.values,
                                   dpos_dv_prime_norm.values), dpos_duv.mask)

        return (minres, maxres)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Surface(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
