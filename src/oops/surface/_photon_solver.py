##########################################################################################
# oops/surface/_photon_solver.py
##########################################################################################

import numpy as np

from polymath              import Qube, Scalar, Vector3
from oops.config           import SURFACE_PHOTONS, LOGGING
from oops.constants        import C
from oops.event            import Event
from oops.frame.frame_     import Frame
from oops.path.path_       import Path

DEBUG = False           # True to log iteration convergence steps

##########################################################################################
# By line of sight
##########################################################################################

def photon_to_event(self, arrival, *, derivs=False, guess=None, antimask=None, quick=None,
                    converge=None):
    """Photon departure from this surface, given arrival and line of sight.

    Parameters:
        arrival (Event): The event of a photon's arrival. Its `arr` attribute must be
            filled in with the Vector3 direction of incoming photons.
        derivs (bool, optional): True to propagate derivatives of the `arrival` position
            and line of sight into the returned event. Derivatives with respect to time
            are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, arrival_event)`, where:

        * `surface_event`: The Event on the surface that matches the light travel time
          to the `arrival`. It is defined in the frame of the Surface and relative to the
          Surface's origin.
        * `arrival_event`: A copy of the given `arrival`, with the photon travel time
          filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon
          from this Surface and `dep_lt` (Scalar) is the positive light travel time to
          `arrival_event`.
        * In `arrival_event`, `arr_lt` (Scalar) is the negative light travel time from
          `surface_event`.
    """

    return _solve_photon_by_los(self, arrival, -1, derivs=derivs, guess=guess,
                                antimask=antimask, quick=quick, converge=converge)


def photon_from_event(self, departure, *, derivs=False, guess=None, antimask=None,
                      quick=None, converge=None):
    """Photon arrival at this surface, given departure and line of sight.

    Parameters:
        departure (Event): The event of a photon's departure. Its `dep` attribute must be
            filled in with the Vector3 direction of outgoing photons.
        derivs (bool, optional): True to propagate derivatives of the `departure` position
            and line of sight into the returned event. Derivatives with respect to time
            are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, departure_event)`, where:

        * `surface_event`: The Event on the surface that matches the light travel time
          from the `departure`. It is defined in the frame of the Surface and relative to
          the Surface's origin.
        * `departure_event`: A copy of the given `departure`, with the photon travel time
          filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `surface_event`, `arr` (Vector3) is the direction of the incoming photon
          and `arr_lt` (Scalar) is the negative light travel time from `departure_event`.
        * In `departure_event`, `dep_lt` (Scalar) is the positive light travel time to
          `surface_event`.
    """

    return _solve_photon_by_los(self, departure, 1, derivs=derivs, guess=guess,
                                antimask=antimask, quick=quick, converge=converge)


def _solve_photon_by_los(self, link, sign, *, derivs=False, guess=None, antimask=None,
                         quick=None, converge=None):
    """Solve for a photon surface intercept from event and line of sight.

    Parameters:
        link (Event): The link event of a photon's arrival or departure.
        sign (int): -1 to return earlier events, corresponding to photons departing from
            the surface and arriving later at the link. +1 to return later events,
            corresponding to photons departing from the link and arriving later at the
            surface.
        derivs (bool, optional): True to propagate derivatives of the link position and
            line of sight into the returned event. Derivatives with respect to time are
            always retained.
        guess (Scalar, optional): An initial guess to use as the event time for the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, link_event)`, where:

        * `surface_event`: The Event on the surface that matches the light travel time
          to or from the `link`. It is defined in the frame of the Surface and relative to
          the Surface's origin.
        * `link_event`: A copy of the given `link`, with the photon travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * If `sign` is negative:
            - In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon
              from this Surface and `dep_lt` (Scalar) is the positive light travel time to
              `link_event`.
            - In `link_event`, `arr_lt` (Scalar) is the negative light travel time from
              `surface_event`.

        * If `sign` is positive:
            - In `surface_event`, `arr` (Vector3) is the direction of the incoming photon
              and `arr_lt` (Scalar) is the negative light travel time from `link_event`.
            - In `link_event`, `dep_lt` (Scalar) is the positive light travel time to
              `surface_event`.
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
    if quick is None:
        quick = {}
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
        return _fully_masked_result(self, link_with_derivs, link_key, coords=True)

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

    # Iterate to solve for lt and surface time. Convergence is rapid because all speeds
    # are non-relativistic.
    max_dlt = np.inf
    converged = False
    hints = True                    # speeds up some calculations
    for count in range(iters):

        # Quicken the path and frame as soon as the range of surface times indicates that
        # this would be beneficial.
        path_wrt_ssb = path_wrt_ssb.quick_path(surface_time, quick=quick)
        frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time, quick=quick)
        # Below, we still pass quick along, because a Path or Frame subclass that does not
        # use QuickPaths or QuickFrames itself might be built upon one that does.

        # Locate the intercept points relative to the origin in SSB/J2000, using the
        # current surface time
        origin_wrt_ssb = path_wrt_ssb.event_at_time(surface_time, quick=quick).pos
        cept_in_j2000 = (obs_wrt_ssb - origin_wrt_ssb) + lt * los_in_j2000

        # Rotate into the surface-fixed frame
        surface_xform = frame_wrt_j2000.transform_at_time(surface_time,
                                                          quick=quick)
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

        if LOGGING.surface_iterations or DEBUG:
            LOGGING.convergence(f'{type(self).__name__}._solve_photon_by_los: '
                                f'iter={count+1}; change[s]={max(max_dlt, 0.):.6g}')

        if max_dlt <= precision:        # converged or fully masked
            converged = True
            break

        if max_dlt >= prev_max_dlt:     # failure to converge
            break

        # Re-evaluate the surface time
        surface_time = link.time + lt

    # END OF LOOP

    if not converged:
        LOGGING.warn('Surface._solve_photon_by_los did not converge;',
                     f'iter={count+1}; change={max_dlt:.6g}')

    # One last iteration with derivatives included
    surface_time = link.time + lt

    if link is not link_with_derivs:
        link = link_with_derivs
        link = link.shrink(antimask)
        link_wrt_ssb = link.wrt_ssb(derivs=True, quick=quick)

    obs_wrt_ssb = link_wrt_ssb.state
    los_in_j2000 = link_wrt_ssb.get_subfield(link_key).with_norm(C)

    origin_wrt_ssb = path_wrt_ssb.event_at_time(surface_time, quick=quick).state
    cept_in_j2000 = (obs_wrt_ssb - origin_wrt_ssb) + lt * los_in_j2000

    surface_xform = frame_wrt_j2000.transform_at_time(surface_time, quick=quick)
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
    new_mask = (lt.values * sign < 0.) | (lt.values == lt_min) | (lt.values == lt_max)
    if np.any(new_mask):
        lt = lt.remask_or(new_mask)

    # If the link is entirely masked, return masked results
    if max_dlt < 0. or np.all(surface_time.mask):
        return _fully_masked_result(self, link_with_derivs, link_key, coords=True)

    # Create the surface event in its own frame

    # The intercept event with respect to the surface has a time-derivative due to the
    # rate of change of the line of sight. However, THIS IS NOT A PHYSICAL VELOCITY. To
    # define the surface event properly, we need to remove the time derivative of
    # cept_wrt_surface. We assign it a new name d_dT to distinguish it from d_dt.

    event_state = cept_in_frame.rename_deriv('t', 'T', method='add')
    event_time  = surface_time.rename_deriv('t', 'T', method='add')
    surface_event = Event(event_time, event_state, self.origin, self.frame)

    # Subfields are calculated using the original cept_in_frame, so these attributes will
    # have correct time-derivatives. This is OK because these time-derivatives are not
    # physical velocities.

    perp = self.normal(cept_in_frame, time=surface_time, derivs=True)
    vflat = self.velocity(cept_in_frame, time=surface_time)
    surface_event.insert_subfield('perp', perp)
    surface_event.insert_subfield('vflat', vflat)
    surface_event.insert_subfield(surface_key, los_in_frame.unit())
    surface_event.insert_subfield(surface_key + '_lt', -lt)

    # Fill in coordinate subfields
    obs_in_frame = cept_in_frame - lt * los_in_frame
    coords = self.coords_from_vector3(cept_in_frame, obs=obs_in_frame,
                                      time=surface_time,
                                      axes=3, derivs=True, hints=hints)
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
    """An entirely masked result, for a link event with nothing left unmasked.

    Parameters:
        link (Event): The link event, whose shape and derivatives the returned Events
            match.
        link_key (str): "arr" if the link event holds the photon's arrival; "dep" if it
            holds the photon's departure.
        coords (bool, optional): True to include masked `coord1`, `coord2`, and `coord3`
            subfields in the returned surface event.

    Returns:
        tuple[Event, Event]: `(surface_event, link_event)`, both entirely masked.
    """

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
        vector.insert_deriv(key, Vector3.ones(link.shape, denom=denom, mask=True))
        scalar.insert_deriv(key, Scalar.ones(link.shape, denom=denom, mask=True))

    # Add link key attributes for the new, masked link
    new_link = link.as_all_masked()
    new_link = new_link.replace(link_key, vector, link_key + '_lt', scalar)

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

##########################################################################################
# By coordinates at the surface
##########################################################################################

def photon_to_coords(self, arrival, coords, *, derivs=False, guess=None, antimask=None,
                     quick=None, converge=None):
    """Photon departure event at the specified surface coordinates.

    Parameters:
        arrival (Event): The event of a photon's arrival.
        coords (tuple[Scalar, ...]): Two or three coordinate values defining locations at
            or near the surface.
        derivs (bool, optional): True to propagate derivatives of the `arrival` position
            and line of sight into the returned event. Derivatives with respect to time
            are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, arrival_event)`, where:

        * `surface_event`: The Event at the surface `coords` that matches the light travel
          time to the `arrival`. It is defined in the frame of the Surface and relative to
          the Surface's origin.
        * `arrival_event`: A copy of the given `arrival`, with the photon line of sight
          and light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon
          from this Surface and `dep_lt` (Scalar) is the positive light travel time to
          `arrival_event`.
        * In `arrival_event`, `arr` (Vector3) is the direction of the incoming photon from
          this Surface and `arr_lt` (Scalar) is the negative light travel time from
          `surface_event`.
    """

    return _solve_photon_by_coords(self, arrival, coords, -1, derivs=derivs, guess=guess,
                                   antimask=antimask, quick=quick, converge=converge)


def photon_from_coords(self, departure, coords, *, derivs=False, guess=None,
                       antimask=None, quick=None, converge=None):
    """Photon arrival event at the specified surface coordinates.

    Parameters:
        departure (Event): The event of a photon's departure.
        coords (tuple[Scalar, ...]): Two or three coordinate values defining locations at
            or near the surface.
        derivs (bool, optional): True to propagate derivatives of the `departure` position
            and line of sight into the returned event. Derivatives with respect to time
            are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, departure_event)`, where:

        * `surface_event`: The Event at the surface `coords` that matches the light travel
          time from the `departure`. It is defined in the frame of the Surface and
          relative to the Surface's origin.
        * `departure_event`: A copy of the given `departure`, with the photon line of
          sight and light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `surface_event`, `arr` (Vector3) is the direction of the incoming photon
          and `arr_lt` (Scalar) is the negative light travel time from `departure_event`.
        * In `departure_event`, `dep` (Vector3) is the direction of the outgoing photon to
          this Surface and `dep_lt` (Scalar) is the positive light travel time to
          `surface_event`.
    """

    return _solve_photon_by_coords(self, departure, coords, 1, derivs=derivs, guess=guess,
                                   antimask=antimask, quick=quick, converge=converge)


def _solve_photon_by_coords(self, link, coords, sign, *, derivs=False, guess=None,
                            antimask=None, quick=None, converge=None):
    """Solve for a photon surface intercept from event and coordinates.

    Parameters:
        link (Event): The link event of a photon's arrival or departure.
        coords (tuple[Scalar, ...]): Two or three coordinate values defining locations at
            or near the surface.
        sign (int): -1 to return earlier events, corresponding to photons departing from
            the surface and arriving later at the link. +1 to return later events,
            corresponding to photons departing from the link and arriving later at the
            surface.
        derivs (bool, optional): True to propagate derivatives of the `link` position and
            coordinates into the returned event. Derivatives with respect to time are
            always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, link_event)`, where:

        * `surface_event`: The Event at the surface `coords` that matches the light travel
          time to or from the `link`. It is defined in the frame of the Surface and
          relative to the Surface's origin.
        * `link_event`: A copy of the given `link`, with the photon line of sight and
          light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * If `sign` is negative:
            - In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon
              from this Surface and `dep_lt` (Scalar) is the positive light travel time to
              `link_event`.
            - In `link_event`, `arr` (Vector3) is the direction of the incoming photon
              from this Surface and `arr_lt` (Scalar) is the negative light travel time
              from `surface_event`.

        * If `sign` is positive:
            - In `surface_event`, `arr` (Vector3) is the direction of the incoming photon
              and `arr_lt` (Scalar) is the negative light travel time from `link_event`.
            - In `link_event`, `dep` (Vector3) is the direction of the outgoing photon to
              this Surface and `dep_lt` (Scalar) is the positive light travel time to
              `surface_event`.
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
    if quick is None:
        quick = {}
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
        return _fully_masked_result(self, link, link_key)

    # Shrink the event. The coordinates are indexed alongside it, so they have to be
    # shrunk by the same antimask; a shapeless coordinate is left as it is.
    unshrunk_link = link
    link = link.shrink(antimask)
    coords = tuple(Scalar.as_scalar(coord).shrink(antimask) for coord in coords)

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

    # For a non-virtual surface whose shape does not vary, pos_wrt_origin is fixed and
    # can be evaluated once. A time-dependent surface has to be re-evaluated inside the
    # loop, because surface_time changes with every iteration.
    if not self.IS_VIRTUAL and not self.IS_TIME_DEPENDENT:
        pos_wrt_origin_frame = self.vector3_from_coords(coords, time=surface_time,
                                                        derivs=True)

    # Iterate to solve for lt. Convergence is rapid because all speeds are
    # non-relativistic.
    max_dlt = np.inf
    converged = False
    for count in range(iters+1):

        # Quicken the path and frame as soon as the range of surface times indicates that
        # this would be beneficial.
        path_wrt_ssb = path_wrt_ssb.quick_path(surface_time, quick=quick)
        frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time, quick=quick)
        # Below, we still pass quick along, because a Path or Frame subclass that does not
        # use QuickPaths or QuickFrames itself might be built upon one that does.

        # Evaluate the observer position relative to the current surface
        origin_wrt_ssb_then = path_wrt_ssb.event_at_time(surface_time, quick=quick).state
        obs_wrt_origin_j2000 = obs_wrt_ssb_now - origin_wrt_ssb_then

        # Locate the coordinate position relative to the current surface
        surface_xform = frame_wrt_j2000.transform_at_time(surface_time, quick=quick)
        if self.IS_VIRTUAL:
            obs_wrt_origin_frame = surface_xform.rotate(obs_wrt_origin_j2000, derivs=True)
            pos_wrt_origin_frame = self.vector3_from_coords(coords,
                                                            obs=obs_wrt_origin_frame,
                                                            time=surface_time,
                                                            derivs=True)
        elif self.IS_TIME_DEPENDENT:
            pos_wrt_origin_frame = self.vector3_from_coords(coords, time=surface_time,
                                                            derivs=True)

        # Locate the coordinate position in J2000
        pos_wrt_origin_j2000 = surface_xform.unrotate(pos_wrt_origin_frame, derivs=True)

        # Update the light travel time
        los_in_j2000 = pos_wrt_origin_j2000 - obs_wrt_origin_j2000
        new_lt = los_in_j2000.norm() / signed_c
        new_lt = new_lt.clip(lt_min, lt_max, remask=False)
        dlt = new_lt - lt
        lt = new_lt

        # Test for convergence
        prev_max_dlt = max_dlt
        max_dlt = abs(dlt).max(builtins=True, masked=-1.)

        if LOGGING.surface_iterations or DEBUG:
            LOGGING.convergence('Surface._solve_photon_by_coords',
                                f'iter={count+1}; change={max_dlt:.6g}')

        if max_dlt <= precision:
            converged = True
            break

        if max_dlt >= prev_max_dlt:
            break

        # Re-evaluate the surface time
        surface_time = link.time + lt

    # END OF LOOP

    if not converged:
        LOGGING.warn('Surface._solve_photon_by_coords did not converge;',
                     f'iter={count+1}; change={max_dlt:.6g}')

    # Update the mask on light time to hide intercepts outside the defined limits
    new_mask = (lt.values * sign < 0.) | (lt.values == lt_min) | (lt.values == lt_max)
    if np.any(new_mask):
        lt = lt.remask_or(new_mask)

    surface_time = link.time + lt

    # If the link is entirely masked, return masked results
    if max_dlt < 0. or np.all(surface_time.mask):
        return _fully_masked_result(self, unshrunk_link, link_key)

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
    new_link = link.replace(link_key + '_j2000', los_in_j2000, link_key + '_lt', lt)

    # Unshrink
    surface_event = surface_event.unshrink(antimask, shape=unshrunk_link.shape)
    new_link = new_link.unshrink(antimask, shape=unshrunk_link.shape)

    return (surface_event, new_link)

##########################################################################################
# Photon Solver based on surface normal and remote event
##########################################################################################

def photon_normal_to_event(self, arrival, *, derivs=False, guess=None, antimask=None,
                           quick=None, converge=None):
    """Photon departure from this surface along the surface normal.

    The arrival event is given; the departing photon is required to have left along the
    normal to the surface.

    This can be used to solve for the sub-observer point on a surface.

    Parameters:
        arrival (Event): The event of a photon's arrival.
        derivs (bool, optional): True to propagate derivatives of the `arrival` position
            and line of sight into the returned event. Derivatives with respect to time
            are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, arrival_event)`, where:

        * `surface_event`: The Event on the surface whose normal points toward the
          `arrival`. It is defined in the frame of the Surface and relative to the
          Surface's origin.
        * `arrival_event`: A copy of the given `arrival`, with the photon line of sight
          and light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon
          from this Surface and `dep_lt` (Scalar) is the positive light travel time to
          `arrival_event`. `perp` (Vector3) is the surface normal and `vflat` (Vector3)
          is the surface velocity at the intercept point. It also carries `coord1`,
          `coord2`, and `coord3` subfields giving the surface coordinates of the normal
          point.
        * In `arrival_event`, `arr` (Vector3) is the direction of the incoming photon
          from this Surface and `arr_lt` (Scalar) is the negative light travel time from
          `surface_event`.
    """

    return _solve_photon_event_normal(self, arrival, -1, derivs=derivs, guess=guess,
                                      antimask=antimask, quick=quick, converge=converge)

def photon_event_to_normal(self, departure, *, derivs=False, guess=None, antimask=None,
                           quick=None, converge=None):
    """Photon arrival at this surface along the surface normal.

    The departure event is given; the arriving photon is required to have arrived along
    the normal to the surface.

    Parameters:
        departure (Event): The event of a photon's departure.
        derivs (bool, optional): True to propagate derivatives of the `departure` position
            and line of sight into the returned event. Derivatives with respect to time
            are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            surface; otherwise None. Should be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, departure_event)`, where:

        * `surface_event`: The Event on the surface whose normal points toward the
          `departure`. It is defined in the frame of the Surface and relative to the
          Surface's origin.
        * `departure_event`: A copy of the given `departure`, with the photon line of
          sight and light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `surface_event`, `arr` (Vector3) is the direction of the incoming photon
          from `departure_event` and `arr_lt` (Scalar) is the negative light travel time
          from `departure_event`. `perp` (Vector3) is the surface normal and `vflat`
          (Vector3) is the surface velocity at the intercept point. It also carries
          `coord1`, `coord2`, and `coord3` Scalars giving the surface coordinates of the
          normal point.
        * In `departure_event`, `dep` (Vector3) is the direction of the outgoing photon
          to this Surface and `dep_lt` (Scalar) is the positive light travel time to
          `surface_event`.
    """

    return _solve_photon_event_normal(self, departure, 1, derivs=derivs, guess=guess,
                                      antimask=antimask, quick=quick, converge=converge)

def _solve_photon_event_normal(self, link, sign, *, derivs=False, guess=None,
                               antimask=None, quick=None, converge=None):
    """The surface intercept event of a photon normal to the surface.

    The event is solved from a remote photon event and the requirement that the apparent
    photon path be normal to the surface.

    Parameters:
        link (Event): The link event of a photon's arrival or departure.
        sign (int): -1 to return earlier events, corresponding to photons departing
            from the surface and arriving later at the link. +1 to return later
            events, corresponding to photons departing from the link and arriving
            later at the surface.
        derivs (bool, optional): True to propagate derivatives of the link position and
            line of sight into the returned event. Derivatives with respect to time are
            always retained.
        guess (Scalar, optional): An initial guess to use as the event time for the
            surface; otherwise None. Should only be used if the event time was already
            returned from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, link_event)`, where:

        * `surface_event`: The Event on the surface whose normal points toward the `link`.
          It is defined in the frame of the Surface and relative to the Surface's origin.
        * `link_event`: A copy of the given `link`, with the photon line of sight and
          light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * If `sign` is negative:
            - In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon
              from this Surface and `dep_lt` (Scalar) is the positive light travel time to
              `link_event`.
            - In `link_event`, `arr` (Vector3) is the direction of the incoming photon
              from this Surface and `arr_lt` (Scalar) is the negative light travel time
              from `surface_event`.

        * If `sign` is positive:
            - In `surface_event`, `arr` (Vector3) is the direction of the incoming photon
              and `arr_lt` (Scalar) is the negative light travel time from `link_event`.
            - In `link_event`, `dep` (Vector3) is the direction of the outgoing photon to
              this Surface and `dep_lt` (Scalar) is the positive light travel time to
              `surface_event`.

        `surface_event` also carries `perp` (Vector3), the surface normal, and `vflat`
        (Vector3), the surface velocity at the intercept point. It also carries `coord1`,
        `coord2`, and `coord3` Scalars giving the surface coordinates of the normal point.
    """

    if self.IS_VIRTUAL:
        raise ValueError('Surface._solve_photon_event_normal does not support '
                         f'virtual surface class {type(self).__name__}')

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
    if quick is None:
        quick = {}
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
        return _fully_masked_result(self, link, link_key)

    # Shrink the event
    unshrunk_link = link
    link = link.shrink(antimask)

    # Define the link event relative to the SSB in J2000
    link_wrt_ssb = link.wrt_ssb(derivs=True, quick=quick)

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
    frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time, quick=quick)

    # Set light travel time limits to avoid a diverging solution
    lt_min = lt.min(builtins=True) - limit
    lt_max = lt.max(builtins=True) + limit

    # Iterate to solve for lt. Convergence is rapid because all speeds are
    # non-relativistic
    max_dlt = np.inf
    converged = False
    hints = True                    # Speeds up some calculations
    p_guess = True                  # Coefficient carried between intercept calls
    for count in range(iters):

        # Evaluate the observer position relative to the current surface
        origin_wrt_ssb_then = origin_wrt_ssb.event_at_time(surface_time).state
        obs_wrt_origin_j2000 = obs_wrt_ssb_now - origin_wrt_ssb_then

        # Locate the coordinate position relative to the current surface
        surface_xform = frame_wrt_j2000.transform_at_time(surface_time)
        obs_wrt_origin_frame = surface_xform.rotate(obs_wrt_origin_j2000, derivs=True)

        # Update the surface intercept. `p_guess` is the coefficient such that
        # cept = obs + p * normal(cept); it is not a light travel time, so it is carried
        # separately and only fed back into the next intercept_normal_to() call.
        (cept_in_frame,
         p_guess, hints) = self.intercept_normal_to(obs_wrt_origin_frame,
                                                    time=surface_time,
                                                    direction=surface_key,
                                                    derivs=True,
                                                    guess=p_guess,
                                                    hints=hints)

        # Update the light travel time from the separation between the intercept and the
        # link event. Distances are frame-independent, so this works in the surface frame.
        new_lt = (cept_in_frame - obs_wrt_origin_frame).norm() / signed_c
        new_lt = new_lt.clip(lt_min, lt_max, remask=False)
        dlt = new_lt - lt
        lt = new_lt

        # Test for convergence
        prev_max_dlt = max_dlt
        max_dlt = abs(dlt).max(builtins=True, masked=-1.)

        if LOGGING.surface_iterations or DEBUG:
            LOGGING.convergence('Surface._solve_photon_event_normal',
                                f'iter={count+1}; change={max_dlt:.6g}')

        if max_dlt <= precision:
            converged = True
            break

        if max_dlt >= prev_max_dlt:
            break

        # Re-evaluate the surface time
        surface_time = link.time + lt

    # END OF LOOP

    if not converged:
        LOGGING.warn('Surface._solve_photon_event_normal did not converge;',
                     f'iter={count+1}; change={max_dlt:.6g}')

    # Update the mask on light time to hide intercepts outside the defined limits
    new_mask = (lt.values * sign < 0.) | (lt.values == lt_min) | (lt.values == lt_max)
    if np.any(new_mask):
        lt = lt.remask_or(new_mask)

    surface_time = link.time + lt

    # If the link is entirely masked, return masked results
    if max_dlt < 0. or np.all(surface_time.mask):
        return _fully_masked_result(self, unshrunk_link, link_key, coords=True)

    # Create the surface event in its own frame

    # The intercept event with respect to the surface has a time-derivative due to the
    # rate of change of the observer position. However, THIS IS NOT A PHYSICAL VELOCITY.
    # To define the surface event properly, we need to remove the time derivative of
    # cept_in_frame. We assign it a new name d_dT to distinguish it from d_dt.

    event_state = cept_in_frame.rename_deriv('t', 'T', method='add')
    event_time  = surface_time.rename_deriv('t', 'T', method='add')
    surface_event = Event(event_time, event_state, self.origin, self.frame)

    # Fill in standard subfields

    # To calculate the time-dependence of other attributes, we need to use the original
    # cept_in_frame in order to give them the correct time- dependence. This is OK because
    # these are not understood to be physical velocities.

    alt_event = Event(surface_time, cept_in_frame, self.origin, self.frame)
    los_in_j2000 = sign * (alt_event.ssb.state - obs_wrt_ssb_now)
    surface_event.insert_subfield(surface_key + '_j2000', los_in_j2000)
    surface_event.insert_subfield(surface_key + '_lt', -lt)

    perp = self.normal(cept_in_frame, time=surface_time, derivs=True)
    vflat = self.velocity(cept_in_frame, time=surface_time)
    surface_event.insert_subfield('perp', perp)
    surface_event.insert_subfield('vflat', vflat)

    # Fill in coordinate subfields
    coords = self.coords_from_vector3(cept_in_frame, obs=obs_wrt_origin_frame,
                                      time=surface_time, axes=3, derivs=True,
                                      hints=hints)
    surface_event.insert_subfield('coord1', coords[0])
    surface_event.insert_subfield('coord2', coords[1])
    surface_event.insert_subfield('coord3', coords[2])

    # Save the hints if any
    if hints is not True:
        surface_event.insert_subfield('hints', hints)

    # Construct the updated link_event
    new_link = link.replace(link_key + '_j2000', los_in_j2000, link_key + '_lt', lt)

    # Unshrink
    surface_event = surface_event.unshrink(antimask, shape=unshrunk_link.shape)
    new_link = new_link.unshrink(antimask, shape=unshrunk_link.shape)

    return (surface_event, new_link)

##########################################################################################
# Photon Solver based on surface normal event and remote path
##########################################################################################

def photon_path_to_normal(self, time, path, *, derivs=False, guess=None, antimask=None,
                          quick=None, converge=None):
    """Photon departure event from a remote path, normal to this surface.

    The photon is required to arrive at this surface at the specified time, along a
    surface normal.

    This can be used to solve for the sub-solar point on a surface.

    Parameters:
        time (Scalar): The time of the photon event at the surface, in seconds TDB.
        path (Path): The remote Path from which the photon departed.
        derivs (bool, optional): True to propagate derivatives of the path position into
            the returned events. Derivatives with respect to time are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            remote path; otherwise None. Should be used if the event time was already
            returned from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, path_event)`, where:

        * `surface_event`: The Event on the surface whose normal points toward `path`. It
          is defined in the frame of the Surface and relative to the Surface's origin.
        * `path_event`: The Event on `path` from which the photon departed.

    Notes:
        The iteration applies light-time correction but no stellar aberration, so
        the photon directions below are actual rather than apparent.

        These subfields are defined in the returned Events:

        * In `surface_event`, `arr` (Vector3) is the direction of the incoming photon,
          which is the surface normal, and `arr_lt` (Scalar) is the negative light
          travel time from `path_event`. `perp` (Vector3) is the surface normal and
          `vflat` (Vector3) is the surface velocity at the intercept point. It also
          carries `coord1`, `coord2`, and `coord3` Scalars giving the surface coordinates
          of the normal point.
        * In `path_event`, `dep` (Vector3) is the direction of the outgoing photon to this
          Surface and `dep_lt` (Scalar) is the positive light travel time to
          `surface_event`.
    """

    return _solve_photon_path_normal(self, time, path, -1, derivs=derivs, guess=guess,
                                     antimask=antimask, quick=quick, converge=converge)


def photon_normal_to_path(self, time, path, *, derivs=False, guess=None, antimask=None,
                          quick=None, converge=None):
    """Photon arrival event at a remote path, normal to this surface.

    The photon is required to have departed this surface at the specified time, along a
    surface normal.

    Parameters:
        time (Scalar): The time of the photon event at the surface, in seconds TDB.
        path (Path): The remote Path at which the photon arrived.
        derivs (bool, optional): True to propagate derivatives of the path position into
            the returned events. Derivatives with respect to time are always retained.
        guess (Scalar, optional): An initial guess to use as the event time at the
            remote path; otherwise None. Should be used if the event time was already
            returned from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, path_event)`, where:

        * `surface_event`: The Event on the surface whose normal points toward `path`. It
          is defined in the frame of the Surface and relative to the Surface's origin.
        * `path_event`: The Event on `path` at which the photon arrived.

    Notes:
        The iteration applies light-time correction but no stellar aberration, so
        the photon directions below are actual rather than apparent.

        These subfields are defined in the returned Events:

        * In `surface_event`, `dep` (Vector3) is the direction of the outgoing photon,
          which is the surface normal, and `dep_lt` (Scalar) is the positive light
          travel time to `path_event`. `perp` (Vector3) is the surface normal and `vflat`
          (Vector3) is the surface velocity at the intercept point.  It also carries
          `coord1`, `coord2`, and `coord3` Scalars giving the surface coordinates of the
          normal point.
        * In `path_event`, `arr` (Vector3) is the direction of the incoming photon from
          this Surface and `arr_lt` (Scalar) is the negative light travel time from
          `surface_event`.
    """

    return _solve_photon_path_normal(self, time, path, 1, derivs=derivs, guess=guess,
                                     antimask=antimask, quick=quick, converge=converge)


def _solve_photon_path_normal(self, time, path, sign, *, derivs=False, guess=None,
                              antimask=None, quick=None, converge=None):
    """Solve for a photon surface intercept based on remote path and local surface normal.

    Parameters:
        time (Scalar): Time at the surface for the photon event.
        path (Path): Remote path for the event associated with the photon's travel.
        sign (int): -1 to return earlier path events, corresponding to photons departing
            from the path and arriving later at the surface. +1 to return later path
            events, corresponding to photons departing from the surface and arriving later
            at the path.
        derivs (bool, optional): True to propagate derivatives of the path position into
            the returned events. Derivatives with respect to time are always retained.
        guess (Scalar, optional): An initial guess to use as the event time for the path;
            otherwise None. Should only be used if the event time was already returned
            from a similar calculation.
        antimask (numpy.ndarray or bool, optional): A boolean filter to be applied to
            event times and positions. Only the indices where antimask=True will be used
            in the solution.
        quick (dict, optional): To override the configured default parameters for
            QuickPaths and QuickFrames; False to disable the use of QuickPaths and
            QuickFrames. The default configuration is defined in config.py.
        converge (dict, optional): Parameters to override the configured default
            convergence parameters. The default configuration is defined in config.py.
            Convergence parameters are as follows:

            * `max_iterations` (int): The maximum number of iterations of Newton's method
              to perform. It should almost never need to be > 6.
            * `dlt_precision` (float): Iteration stops when the largest change in light
              travel time between one iteration and the next falls below this threshold
              (in seconds).
            * `dlt_limit` (float): The maximum allowed absolute value of the change in
              light travel time from the nominal range calculated initially. Changes in
              light travel with absolute values larger than this limit are clipped. This
              prevents the divergence of the solution in some cases.

    Returns:
        tuple[Event, Event]: `(surface_event, path_event)`, where:

        * `surface_event`: The Event on the surface whose normal points toward `path`. It
          is defined in the frame of the Surface and relative to the Surface's origin.
        * `path_event`: The Event on the remote `path`.

    Notes:
        These subfields are defined in the returned Events:

        * If `sign` is negative:
            - In `surface_event`, `arr` (Vector3) is the direction of the incoming
              photon, which is the surface normal, and `arr_lt` (Scalar) is the negative
              light travel time from `path_event`.
            - In `path_event`, `dep` (Vector3) is the direction of the outgoing photon to
              this Surface and `dep_lt` (Scalar) is the positive light travel time to
              `surface_event`.

        * If `sign` is positive:
            - In `surface_event`, `dep` (Vector3) is the direction of the outgoing
              photon, which is the surface normal, and `dep_lt` (Scalar) is the positive
              light travel time to `path_event`.
            - In `path_event`, `arr` (Vector3) is the direction of the incoming photon
              from this Surface and `arr_lt` (Scalar) is the negative light travel time
              from `surface_event`.

        `surface_event` also carries `perp` (Vector3), the surface normal, and `vflat`
        (Vector3), the surface velocity at the intercept point. It also carries `coord1`,
        `coord2`, and `coord3` Scalars giving the surface coordinates of the normal point.
    """

    if self.IS_VIRTUAL:
        raise ValueError('Surface._solve_photon_path_normal does not support '
                         f'virtual surface class {type(self).__name__}')

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
    if quick is None:
        quick = {}
    if isinstance(quick, dict):
        quick = quick.copy()
        quick['path_time_extension'] = limit
        quick['frame_time_extension'] = limit

    # Interpret the sign. Note that `sign` has the opposite meaning here from the other
    # solvers in this module, because it describes the photon relative to the remote path
    # rather than relative to the surface; `signed_c` is negated to match.
    signed_c = -sign * C
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
        return _fully_masked_result(self, unshrunk_remote, remote_key)

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
        origin_event = Event(surface_time.wod, Vector3.ZERO, self.origin, self.frame)
        (path_event, _) = path._solve_photon(origin_event, sign, quick=quick,
                                             converge=converge)
        path_time = path_event.time.wod
    else:
        path_time = guess

    lt = surface_time.wod - path_time

    # Lock down the surface origin and frame relative to SSB/J2000
    surface_xform = self.frame.transform_at_time(surface_time)
    origin_wrt_ssb = self.origin.event_at_time(surface_time).state

    # Define the path relative to SSB/J2000 and quicken
    path_wrt_ssb = path.wrt(Path.SSB, Frame.J2000)
    path_wrt_ssb = path_wrt_ssb.quick_path(path_time, quick=quick)

    # Iterate to solve for lt. Convergence is rapid because all speeds are
    # non-relativistic.
    max_dlt = np.inf
    converged = False
    hints = True                # Speeds up some calculations
    p_guess = True              # Coefficient carried between intercept_normal_to calls
    for count in range(iters):

        # Locate position relative to origin in SSB/J2000
        pos_wrt_origin_j2000 = (path_wrt_ssb.event_at_time(path_time).state
                                - origin_wrt_ssb)

        # Locate position relative to origin in surface frame
        pos_wrt_origin_frame = surface_xform.rotate(pos_wrt_origin_j2000, derivs=True)

        # Update the surface intercept. `p_guess` is the coefficient such that
        # cept = pos + p * normal(cept); it is not a light travel time, so it is carried
        # separately and only fed back into the next intercept_normal_to() call.
        (cept_in_frame,
         p_guess, hints) = self.intercept_normal_to(pos_wrt_origin_frame,
                                                    time=surface_time,
                                                    direction=surface_key,
                                                    derivs=True,
                                                    guess=p_guess,
                                                    hints=hints)

        # Update the light travel time from the separation between the intercept and the
        # remote path. Distances are frame-independent, so the surface frame will do.
        new_lt = (pos_wrt_origin_frame - cept_in_frame).norm() / signed_c
        dlt = new_lt - lt
        lt = new_lt

        # Test for convergence
        prev_max_dlt = max_dlt
        max_dlt = abs(dlt).max(builtins=True, masked=-1.)

        if LOGGING.surface_iterations or DEBUG:
            LOGGING.convergence('Surface._solve_photon_path_normal',
                                f'iter={count+1}; change={max_dlt:.6g}')

        if max_dlt <= precision:
            converged = True
            break

        if max_dlt >= prev_max_dlt:
            break

        # Re-evaluate the path time. `lt` is the offset of the surface event relative
        # to the path event, so the path event lies on the opposite side.
        path_time = surface_time - lt

    #### END OF LOOP

    if not converged:
        LOGGING.warn('Surface._solve_photon_path_normal did not converge;',
                     f'iter={count+1}; change={max_dlt:.6g}')

    # If the result is entirely masked, return masked results
    if max_dlt < 0. or np.all(path_time.mask):
        # This is a fake, fully masked link
        vec = Vector3.ZERO.broadcast_to(path_time.shape)
        link = Event(path_time.remask(True), (vec, vec), path, frame=Frame.J2000)
        return _fully_masked_result(self, link, remote_key, coords=False)

    # Create the surface event in its own frame

    # The intercept event with respect to the surface has a time-derivative due to the
    # rate of change of the observer position. However, THIS IS NOT A PHYSICAL VELOCITY.
    # To define the surface event properly, we need to remove the time derivative of
    # cept_in_frame. We assign it a new name d_dT to distinguish it from d_dt.

    event_state = cept_in_frame.rename_deriv('t', 'T', method='add')

    # Unlike the other solvers, the surface time here is an input rather than a
    # solution, so it carries no spurious time-derivative to demote.
    if 't' in surface_time.derivs:
        event_time = surface_time.rename_deriv('t', 'T', method='add')
    else:
        event_time = surface_time

    surface_event = Event(event_time, event_state, self.origin, self.frame)

    # Subfields are calculated using the original cept_in_frame, so these attributes will
    # have correct time-derivatives. This is OK because these time-derivatives are not
    # physical velocities.

    # The iteration above applies light-time correction but no stellar aberration, so the
    # normal is the actual photon direction, not the apparent one. Inserting it under the
    # plain key rather than '_ap' keeps the '_j2000' value read back below a pure rotation
    # into J2000; inserting it as apparent would make that read-back remove an aberration
    # that was never applied.
    normal = self.normal(cept_in_frame, time=surface_time, derivs=True)
    surface_event.insert_subfield(surface_key, normal)
    surface_event.insert_subfield(surface_key + '_lt', -lt)
    surface_event.insert_subfield('perp', normal)
    surface_event.insert_subfield('vflat',
                                  self.velocity(cept_in_frame, time=surface_time))

    # Fill in coordinate subfields
    coords = self.coords_from_vector3(cept_in_frame, obs=cept_in_frame,
                                      time=surface_time,
                                      axes=3, derivs=True, hints=hints)
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

##########################################################################################
