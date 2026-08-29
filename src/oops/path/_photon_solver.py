##########################################################################################
# oops/path/_photon_solver.py
##########################################################################################

import numpy as np

from polymath              import Qube, Scalar, Vector3
from oops.config           import PATH_PHOTONS, LOGGING
from oops.frame.frame_     import Frame
from oops.path.path_       import Path
import oops.constants as constants


def photon_to_event(self, arrival, *, derivs=False, guess=None, antimask=None, quick=None,
                    converge=None):
    """The photon departure event from this Path to match the arrival event.

    Parameters:
        arrival (Event): The Event of a photon's arrival.
        derivs (bool, optional): True to propagate derivatives of the `arrival` position
            into the returned Events. The time derivative is always retained.
        guess (Scalar, array-like, or float, optional): An initial guess to use as the
            event time along this Path; otherwise None. Should be provided if the event
            time was already returned from a similar calculation.
        antimask (ndarray or bool, optional): A boolean array to be applied to event times
            and positions. Only the indices where antimask=True will be used in the
            solution.
        quick (dict or bool, optional): An optional dictionary of parameter values to use
            as overrides to the configured default QuickPath and QuickFrame parameters;
            use False to disable the use of QuickPaths and QuickFrames. The default quick
            dictionary is defined in config.py.
        converge (dict, optional): An optional dictionary of parameters to override the
            configured default convergence parameters. The default configuration is
            defined in config.py. Convergence parameters are as follows:

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
        tuple[Event, Event]: (`path_event`, `arrival_event`).

        * `path_event`: The Event on this Path that matches the light travel time to
          `arrival`. This Event always has position (0,0,0) on the Path, and it holds the
          departing photon's line of sight and light travel time.
        * `arrival_event`: A copy of the given `arrival` Event, with the photon's arriving
          line of sight and light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `path_event`, `dep` (Vector3) is the direction of the outgoing photon from
          this Path and `dep_lt` (Scalar) is the positive light travel time to
          `arrival_event`.
        * In `arrival_event`, `arr` (Vector3) is the direction of the incoming photon and
          `arr_lt` (Scalar) is the negative light travel time from `path_event`.
    """

    return self._solve_photon(arrival, -1, derivs=derivs, guess=guess, antimask=antimask,
                              quick=quick, converge=converge)


def photon_from_event(self, departure, *, derivs=False, guess=None, antimask=None,
                      quick=None, converge=None):
    """The photon arrival event at this Path to match the departure event.

    Parameters:
        departure (Event): The Event of a photon's departure.
        derivs (bool, optional): True to propagate derivatives of the `departure` position
            into the returned Events. The time derivative is always retained.
        guess (Scalar, array-like, or float, optional): An initial guess to use as the
            event time along this Path; otherwise None. Should be provided if the event
            time was already returned from a similar calculation.
        antimask (ndarray or bool, optional): A boolean array to be applied to event times
            and positions. Only the indices where antimask=True will be used in the
            solution.
        quick (dict or bool, optional): An optional dictionary of parameter values to use
            as overrides to the configured default QuickPath and QuickFrame parameters;
            use False to disable the use of QuickPaths and QuickFrames. The default quick
            dictionary is defined in config.py.
        converge (dict, optional): An optional dictionary of parameters to override the
            configured default convergence parameters. The default configuration is
            defined in config.py. Convergence parameters are as follows:

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
        tuple[Event, Event]: (`path_event`, `departure_event`).

        * `path_event`: The Event on this Path that matches the light travel time from
          `departure`. This Event always has position (0,0,0) on the path, and it holds
          the arriving photon's line of sight and light travel time.
        * `departure_event`: A copy of the given `departure` Event, with the photon's
          departing line of sight and light travel time filled in.

    Notes:
        These subfields are defined in the returned Events:

        * In `path_event`, `arr` (Vector3) is the direction of the incoming photon from
          `departure_event` and `arr_lt` (Scalar) is the negative light travel time.
        * In `departure_event`, `dep` (Vector3) is the direction of the outgoing photon
          and `dep_lt` (Scalar) is the positive light travel time to `path_event`.
    """

    return self._solve_photon(departure, 1, derivs=derivs, guess=guess, antimask=antimask,
                              quick=quick, converge=converge)


def _solve_photon(self, link, sign, *, derivs=False, guess=None, antimask=None,
                  quick=None, converge=None):
    """Solve for a photon arrival or departure event on this path.

    Parameters:
        link (Event): The Event of a photon's arrival or departure.
        sign (int): -1 to return earlier Events, corresponding to photons departing from
            this Path and arriving at the Event; +1 to return later Events, corresponding
            to photons arriving at this Path after departing from the Event.
        derivs (bool, optional): True to propagate derivatives of the link position into
            the returned event. The time derivative is always retained.
        guess (Scalar, array-like, or float, optional): An initial guess to use as the
            event time along this Path; otherwise None. Should be provided if the event
            time was already returned from a similar calculation.
        antimask (ndarray or bool, optional): A boolean array to be applied to event times
            and positions. Only the indices where antimask=True will be used in the
            solution.
        quick (dict or bool, optional): An optional dictionary of parameter values to use
            as overrides to the configured default QuickPath and QuickFrame parameters;
            use False to disable the use of QuickPaths and QuickFrames. The default quick
            dictionary is defined in config.py.
        converge (dict, optional): An optional dictionary of parameters to override the
            configured default convergence parameters. The default configuration is
            defined in config.py. Convergence parameters are as follows:

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
        tuple[Event, Event]: (`path_event`, `link_event`).

        * `path_event`: The Event on this Path that matches the light travel time from the
          `link` event. This Event always has position (0,0,0) on the path.
        * `link_event`: A copy of the given `link` Event, with the photon arrival or
          departure line of sight and light travel time filled in.
    """

    # Internal function to return an entirely masked result
    def fully_masked_results():
        vector3 = Vector3(np.ones(original_link.shape + (3,)), True)
        scalar = Scalar(vector3.values[...,0], True)

        if derivs:
            scalar.insert_deriv('t', Scalar(1., True), override=True)
            scalar.insert_deriv('los',
                                Scalar(np.ones((1,3)), True, drank=1),
                                override=True)

            vector3.insert_deriv('t', Vector3((1,1,1), True), override=True)
            vector3.insert_deriv('los',
                                 Vector3(np.ones((3,3)), True, drank=1),
                                 override=True)

        new_link = original_link.replace(link_key, vector3,
                                         link_key + '_lt', scalar)
        new_link = new_link.as_all_masked()

        path_event = new_link.as_all_masked(origin=self.origin,
                                            frame=self.frame.wayframe)
        path_event = path_event.replace(path_key, vector3,
                                        path_key + '_lt', scalar)

        return (path_event, new_link)

    original_link = link

    # Handle derivatives
    if not derivs:
        link = link.wod     # preserves time-derivatives; removes others

    # Assemble convergence parameters
    if converge:
        defaults = PATH_PHOTONS.__dict__.copy()
        defaults.update(converge)
        converge = defaults
    else:
        converge = PATH_PHOTONS.__dict__

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

    # Iterate to a solution for the light travel time "lt". Define
    #   y = separation_distance(time + lt) - sign * c * lt
    # where lt is negative for earlier linking events and positive for later linking
    # events.
    #
    # Solve for the value of lt at which y = 0, using Newton's method.
    #
    # Approximate the function as linear around the solution:
    #   y[n+1] - y[n] = (lt[n+1] - lt[n]) * dy_dlt
    # Our goal is for the next value of y, y[n+1], to equal zero. Our most recent
    # guess is (lt[n], y[n]).
    #
    # What should we use for lt[n+1]?
    #   lt[n+1] = lt[n] - y[n] / dy_dlt
    #
    # The function y is shown above. Its derivative is
    #   dy_dlt = outward_speed - sign * c

    # Interpret the sign
    signed_c = sign * constants.C
    if sign < 0.:           # photon_to_event case
        path_key = 'dep'    # departure event is on the path, sign < 0, dep_lt < 0
        link_key = 'arr'    # link event holds the photon's arrival, arr_lt > 0
    else:                   # photon_from_event case
        path_key = 'arr'    # arrival event is on the path, sign > 0, arr_lt > 0
        link_key = 'dep'    # link event holds the photon's departure, dep_lt < 0

    # Define the antimask
    if antimask is None:
        antimask = link.antimask
    else:
        antimask = antimask & link.antimask

    # If the link is entirely masked...
    if not np.any(antimask):
        return fully_masked_results()

    # Shrink the event
    link = link.shrink(antimask)

    # Define quantities with respect to SSB in J2000
    link_wrt_ssb = link.wrt_ssb(derivs=derivs, quick=quick)
    path_wrt_ssb = self.wrt(Path.SSB, Frame.J2000)

    # Prepare for iteration, avoiding any derivatives for now
    link_time = link.time.wod
    link_pos_ssb = link_wrt_ssb.pos.wod
    link_vel_ssb = link_wrt_ssb.vel.wod
    link_shape = link.shape

    # Make initial guesses at the path event time
    if guess is not None:
        path_time = Scalar.as_scalar(guess).wod.shrink(antimask)
        lt = path_time - link_time
    else:
        lt = (path_wrt_ssb.event_at_time(link_time, quick=quick).pos.wod
              - link_pos_ssb).norm() / signed_c
        path_time = link_time + lt

    # Set light travel time limits to avoid a diverging solution
    lt_min = (path_time - link_time).min() - limit
    lt_max = (path_time - link_time).max() + limit

    lt_min = lt_min.as_builtin()
    lt_max = lt_max.as_builtin()

    # Broadcast the path_time to encompass the shape of the path, if any
    shape = Qube.broadcasted_shape(path_time, link_shape)
    if path_time.shape != shape:
        path_time = path_time.broadcast_to(shape)

    # Iterate a fixed number of times or until the threshold of error
    # tolerance is reached. Convergence takes just a few iterations.
    max_dlt = np.inf
    prev_lt = None
    converged = False
    for count in range(iters):

        # Quicken the path as soon as the range of times indicates that this would
        # be beneficial. `quick` is still passed below, because a Path that does not
        # use QuickPaths itself might be built upon one that does.
        path_wrt_ssb = path_wrt_ssb.quick_path(path_time, quick=quick)

        # Evaluate the photon's current SSB position based on time
        path_event_ssb = path_wrt_ssb.event_at_time(path_time, quick=quick)
        delta_pos_ssb = path_event_ssb.pos.wod - link_pos_ssb
        delta_vel_ssb = path_event_ssb.vel.wod - link_vel_ssb

        dlt = ((delta_pos_ssb.norm() - lt * signed_c)
               / (delta_vel_ssb.proj(delta_pos_ssb).norm() - signed_c))
        new_lt = (lt - dlt).clip(lt_min, lt_max, remask=False)
        dlt = lt - new_lt

        prev_lt = lt
        lt = new_lt

        # Re-evaluate the path time
        path_time = link_time + lt

        # Test for convergence
        prev_max_dlt = max_dlt
        max_dlt = abs(dlt).max(builtins=True, masked=-1.)

        if LOGGING.surface_iterations:
            LOGGING.performance(f'Path._solve_photon: iter={count+1}; '
                                f'change={max_dlt:.6g}')

        if max_dlt <= precision:
            converged = True
            break

        if max_dlt >= prev_max_dlt:
            break

    # END OF LOOP

    if not converged:
        LOGGING.warn(f'Path._solve_photon did not converge: iter={count+1}; '
                     f'change={max_dlt:.6g}')

    # If the link is entirely masked...
    if max_dlt < 0.:
        return fully_masked_results()

    # Restore derivatives to path_time if necessary
    # This is a repeat of the final iteration, but with derivatives included
    if derivs:
        delta_pos_ssb = path_event_ssb.state - link_wrt_ssb.state
        delta_vel_ssb = path_event_ssb.vel - link_wrt_ssb.vel

        dlt = ((delta_pos_ssb.norm() - prev_lt * signed_c)
               / (delta_vel_ssb.proj(delta_pos_ssb).norm() - signed_c))
        new_lt = (prev_lt - dlt).clip(lt_min, lt_max, remask=False)
        path_time = link.time + new_lt

        # The path_time contains a time derivative due to the motion of the
        # link. We rename this derivative from 't' to 'T' to avoid
        # confusion.
        path_time = path_time.rename_deriv('t', 'T', method='add')

    # Construct the returned event
    path_event_ssb = path_wrt_ssb.event_at_time(path_time, quick=quick)
    link_event_ssb = link_wrt_ssb.copy()

    # Fill in the key subfields
    if sign > 0:
        ray_vector_ssb = (path_event_ssb.state - link_event_ssb.state).as_readonly()
    else:
        ray_vector_ssb = (link_event_ssb.state - path_event_ssb.state).as_readonly()

    lt = ray_vector_ssb.norm(recursive=derivs) / signed_c
    # photon_to_event:   sign < 0, lt < 0, path_key = 'dep', link_key = 'arr'
    # photon_from_event: sign > 0, lt > 0, path_key = 'arr', link_key = 'dep'

    path_event_ssb = path_event_ssb.replace(path_key, ray_vector_ssb,
                                            path_key + '_lt', -lt)
    # photon_to_event:   path event has dep_lt, sign < 0, lt < 0, dep_lt > 0
    # photon_from_event: path event has arr_lt, sign > 0, lt > 0, arr_lt < 0

    # Transform the path event into its origin and frame
    path_event = path_event_ssb.from_ssb(self, self.frame, derivs=derivs, quick=quick)

    # Transform the light ray into the link's frame
    new_link = link.replace(link_key + '_j2000', ray_vector_ssb,
                            link_key + '_lt', lt)
    # photon_to_event:   link event has arr_lt, sign < 0, lt < 0, arr_lt < 0
    # photon_from_event: link event has dep_lt, sign > 0, lt > 0, dep_lt > 0

    # Unshrink
    path_event = path_event.unshrink(antimask)
    new_link = new_link.unshrink(antimask)

    return (path_event, new_link)

##########################################################################################
