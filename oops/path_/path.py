################################################################################
# oops/path/path.py: Abstract class Path and its required subclasses
################################################################################

import numpy as np
import scipy.interpolate as interp
from polymath import *

from oops.config       import QUICK, PATH_PHOTONS, LOGGING
from oops.event        import Event
from oops.frame_.frame import Frame, Wayframe
import oops.constants  as constants

class Path(object):
    """Path is an abstract class that returns an Event (time, position and
    velocity) given a Scalar time. The coordinates are specified in a particular
    frame and relative to another path. All paths are ultimately references to
    the Solar System Barycenter ("SSB") and the J2000 coordinate frame."""

    WAYPOINT_REGISTRY = {}
    PATH_CACHE = {}
    TEMPORARY_PATH_ID = 10000

    ############################################################################
    # Each subclass must override...
    ############################################################################

    def __init__(self):
        """Constructor for a Path object. Every path must have these attributes:

            path_id     the string ID of this Path.
            waypoint    the waypoint that uniquely identifies this path. For
                        registered paths, this is the Waypoint object with
                        the same ID, as it appears in the WAYPOINT_REGISTRY
                        dictionary. If a path is not registered, then its
                        waypoint attribute should point to itself.
            origin      the origin waypoint, relative to which this Path is
                        defined.
            frame       the wayframe identifying the frame used by the event
                        objects returned.
            shape       the shape of this object as a tuple, i.e, the shape of
                        the event returned when a single value of time is passed
                        to event_at_time().
            keys        the set of keys by which this path is cached.

        The primary definition of a path will be assigned these attributes by
        the registry:

            ancestry    a list of Path objects beginning with this one and
                        ending with with the Solar System Barycenter. Each Path
                        in the sequence is the origin of the previous Path:
                            self.ancestry[0] = self.
                            self.ancestry[1] = origin path of self.
                            ...
                            self.ancestry[-1] = SSB in J2000.

            wrt_ssb     a definition of the same path relative to the Solar
                        System Barycenter, in the J2000 coordinate frame.
        """

        pass

    def event_at_time(self, time, quick={}):
        """An Event corresponding to a specified Scalar time on this path.

        Input:
            time        a Scalar time at which to evaluate the path.
            quick       an optional dictionary of parameter values to use as
                        overrides to the configured default QuickPath and
                        QuickFrame parameters; use False to disable the use of
                        QuickPaths and QuickFrames.

        Return:         an event object containing (at least) the time, position
                        and velocity of the path.

        Note that the time and the path are not required to have the same shape;
        standard rules of broadcasting define the shape of the returned Event.
        """

        pass

    @property
    def origin_id(self): return self.origin.path_id

    @property
    def frame_id(self): return self.frame.frame_id

    # string operations
    def __str__(self):
        return (type(self).__name__ + '([' + self.path_id   + ' - ' +
                                             self.origin_id + ']*' +
                                             self.frame_id + ')')

    def __repr__(self): return self.__str__()

    ############################################################################
    # Registry Management
    ############################################################################

    # A path can be registered by an ID string. Any path so registered can be
    # retrieved afterward from the registry using the string. However, it is not
    # necessary to register a path; just set the attribute 'waypoint' to self
    # instead.
    #
    # When an ID is registered for the first time, a Waypoint is constructed and
    # added to the WAYPOINT_REGISTRY, which is a dictionary that returns the
    # Waypoint object associated with any ID.
    #
    # If a path definition is overridden, then a new Waypoint is created and
    # replaces the previous one in the registry. The old path will still exist
    # and can still be used, but it will no longer be accessible from the
    # registry.
    #
    # The PATH_CACHE contains every calculated version of a Path object. This
    # saves us the effort of re-connecting a path (waypoint, origin, frame) each
    # time is is needed. The PATH_CACHE is keyed as follows:
    #       waypoint
    #       (waypoint, origin_waypoint)
    #       (waypoint, origin_waypoint, wayframe)
    #
    # If the key is not a tuple, then this constitutes the primary definition of
    # the path. The origin waypoint must already be in the cache, and the new
    # waypoint cannot be in the cache.

    @staticmethod
    def initialize_registry():
        """Initialize the path registry.

        It is not generally necessary to call this function directly."""

        # The frame registry must be initialized first
        Frame.initialize_registry()

        # After first call, return
        if Path.WAYPOINT_REGISTRY: return

        # Initialize the WAYPOINT_REGISTRY
        Path.WAYPOINT_REGISTRY[None] = Path.SSB
        Path.WAYPOINT_REGISTRY['SSB'] = Path.SSB

        # Initialize the PATH_CACHE
        Path.SSB.keys = {Path.SSB,
                         (Path.SSB, Path.SSB),
                         (Path.SSB, Path.SSB, Frame.J2000)}
        for key in Path.SSB.keys:
            Path.PATH_CACHE[key] = Path.SSB

    @staticmethod
    def reset_registry():
        """Reset the registry to its initial state. Mainly useful for debugging.
        """

        Path.WAYPOINT_REGISTRY.clear()
        Path.PATH_CACHE.clear()
        Path.initialize_registry()

    def register(self, shortcut=None, override=False):
        """Register a Path's definition.

        A shortcut makes it possible to calculate the state of one SPICE body
        relative to another without calculating the states of all the
        intermediate objects. If a shortcut name is given, then this path is
        treated as a shortcut definition. The path is cached under the shortcut
        name and also under the tuple (waypoint, origin_waypoint, wayframe).

        If override is True, then this path will override the definition of any
        previous path with the same name. The old path might still exist but it
        will not be available from the registry.

        If the path ID is None, blank, or begins with '.', it is treated as a
        temporary path and is not registered.
        """

        # Make sure the registry is initialized
        if Path.SSB is None: Path.initialize_registry()

        WAYPOINT_REG = Path.WAYPOINT_REGISTRY
        PATH_CACHE = Path.PATH_CACHE

        id = self.path_id

        # Handle a shortcut
        if shortcut is not None:
            if shortcut in PATH_CACHE: PATH_CACHE[shortcut].keys -= {shortcut}
            PATH_CACHE[shortcut] = self
            self.keys |= {shortcut}

            key = (WAYPOINT_REG[id], self.origin, self.frame)
            if key in PATH_CACHE: PATH_CACHE[key].keys -= {key}
            PATH_CACHE[key] = self
            self.keys |= {key}

            if not hasattr(self, 'waypoint') or self.waypoint is None:
                self.waypoint = WAYPOINT_REG[id]

            return

        # Fill in a temporary name if needed; don't register
        if self.path_id in (None, '', '.'):
            self.path_id = Path.temporary_path_id()
            self.waypoint = self
            return

        # Don't register a name beginning with dot
        if self.path_id.startswith('.'):
            self.waypoint = self
            return

        # Make sure the origin path is registered; raise a KeyError otherwise
        test = WAYPOINT_REG[self.origin.path_id]
        test = PATH_CACHE[self.origin]

        # If the ID is unregistered, insert this as a primary definition
        if id not in WAYPOINT_REG or override:

            # Fill in the ancestry
            origin = Path.as_primary_path(self.origin)
            self.ancestry = [origin] + origin.ancestry

            # Register the Waypoint
            waypoint = Waypoint(id, self.frame, self.shape)
            self.waypoint = waypoint
            WAYPOINT_REG[id] = waypoint

            # Cache the path under three keys
            self.keys = {waypoint,
                         (waypoint, self.origin),
                         (waypoint, self.origin, self.frame)}
            for key in self.keys:
                PATH_CACHE[key] = self

            # Cache the waypoint under two or three keys
            waypoint.keys = {(waypoint, waypoint),
                             (waypoint, waypoint, self.frame),
                             (waypoint, waypoint, Frame.J2000)}
            for key in waypoint.keys:
                PATH_CACHE[key] = waypoint

            # Also define the path with respect to the SSB
            if self.origin == Path.SSB and self.frame == Frame.J2000:
                self.wrt_ssb = self
            else:
                self.wrt_ssb = self.wrt(Path.SSB, Frame.J2000)

                self.wrt_ssb.keys = {(waypoint, Path.SSB, Frame.J2000)}
                if self.origin != Path.SSB:
                    self.wrt_ssb.keys |= {(waypoint, Path.SSB)}

                for key in self.wrt_ssb.keys:
                    PATH_CACHE[key] = self.wrt_ssb

        # Otherwise, just insert secondary definitions
        else:
            if not hasattr(self, 'waypoint') or self.waypoint is None:
                self.waypoint = WAYPOINT_REG[id]

            # Cache (self.waypoint, self.origin); overwrite if necessary
            key = (self.waypoint, self.origin)
            if key in PATH_CACHE:           # remove an old version
                PATH_CACHE[key].keys -= {key}

            PATH_CACHE[key] = self
            self.keys |= {key}

            # Cache (self.waypoint, self.origin, self.frame)
            key = (self.waypoint, self.origin, self.frame)
            if key in PATH_CACHE:           # remove an old version
                PATH_CACHE[key].keys -= {key}

            PATH_CACHE[key] = self
            self.keys |= {key}

    @staticmethod
    def as_path(path):
        """Return a Path object given the ID or the object itself."""

        if path is None: return None

        if isinstance(path, Path): return path

        return Path.WAYPOINT_REGISTRY[path]

    @staticmethod
    def as_primary_path(path):
        """Return the primary definition of a Path object given a path or ID."""

        if path is None: return None

        if not isinstance(path, Path):
            path = Path.WAYPOINT_REGISTRY[path]

        return Path.PATH_CACHE[path.waypoint]

    @staticmethod
    def as_waypoint(path):
        """Return the waypoint given a Path or ID."""

        if path is None: return None

        if isinstance(path, Path):
            return path.waypoint

        return Path.WAYPOINT_REGISTRY[path]

    @staticmethod
    def as_path_id(path):
        """Return a path ID given the object or its ID."""

        if path is None: return None

        if isinstance(path, Path):
            return path.path_id

        return path

    @staticmethod
    def temporary_path_id():
        """Return a temporary path ID. This is assigned once and never re-used.
        """

        while True:
            Path.TEMPORARY_PATH_ID += 1
            path_id = "TEMPORARY_" + str(Path.TEMPORARY_PATH_ID)

            if path_id not in Path.WAYPOINT_REGISTRY:
                return path_id

    def is_registered(self):
        """True if this frame is registered."""

        return (self.path_id in Path.WAYPOINT_REGISTRY)

    ############################################################################
    # Event operations
    ############################################################################

    # These must be defined here and not in Event.py, because that would create
    # a circular dependency in the order that modules are loaded.

    def subtract_from_event(self, event, derivs=True, quick={}):
        """An equivalent Event, but with this path redefining the origin.

        Input:
            event       the event object from which this path is to be
                        subtracted. The path's origin must coincide with the
                        event's origin, and the two objects must use the same
                        frame.
            derivs      True to include derivatives in the attributes of the
                        returned event. The specific derivatives included will
                        depend on the Path subclass and those within the given
                        Event.
            quick       an optional dictionary of parameter values to use as
                        overrides to the configured default QuickPath and
                        QuickFrame parameters; use False to disable the use of
                        QuickPaths and QuickFrames.
        """

        # Check for compatibility
        if self.origin.waypoint != event.origin.waypoint:
            raise ValueError('Events must have a common origin path for ' +
                             'path subtraction')

        if self.frame.wayframe != event.frame.wayframe:
            raise ValueError('Events must share a common frame for path ' +
                             'subtraction')

        # Create the path event
        path_event = self.event_at_time(event.time, quick=quick)

        # Strip derivatives from the given event if necessary
        if not derivs:
            event = event.without_derivs()

        # Subtract
        return Event(event.time, event.state - path_event.state,
                     self, self.frame, **event.subfields)

    def add_to_event(self, event, derivs=True, quick={}):
        """An equivalent event, using the origin of this path as the origin.

        Input:
            event       the event object to which this path is to be added. The
                        path's endpoint must coincide with the event's origin,
                        and the two objects must use the same frame.
            derivs      True to include derivatives in the attributes of the
                        returned event. The specific derivatives included will
                        depend on the Path subclass and the given Event. Time
                        derivatives are always retained.
            quick       an optional dictionary of parameter values to use as
                        overrides to the configured default QuickPath and
                        QuickFrame parameters; use False to disable the use of
                        QuickPaths and QuickFrames.
        """

        # Check for compatibility
        if self.waypoint != event.origin.waypoint:
            raise ValueError("An Event's origin must match this path for " +
                             'path addition')

        if self.frame.wayframe != event.frame.wayframe:
            raise ValueError('Events must share a common frame for path ' +
                             'addition')

        # Create the path event
        path_event = self.event_at_time(event.time, quick=quick)

        # Strip derivatives from the given event if necessary
        if not derivs:
            event = event.without_derivs()

        # Add
        return Event(event.time, event.state + path_event.state,
                     self.origin, event.frame, **event.subfields)

    ############################################################################
    # Photon Solver
    ############################################################################

    def photon_to_event(self, arrival, derivs=False, guess=None,
                              quick={}, converge={}):
        """The photon departure event from this path to match the arrival event.

        See _solve_photon() for details.
        """

        return self._solve_photon(arrival, -1, derivs, guess, quick, converge)

    def photon_from_event(self, departure, derivs=False, guess=None,
                                quick={}, converge={}):
        """The photon arrival event at this path to match the departure event.

        See _solve_photon() for details.
        """

        return self._solve_photon(departure, 1, derivs, guess, quick, converge)

    def _solve_photon(self, link, sign, derivs=False, guess=None,
                            quick={}, converge={}):

        """Solve for a photon arrival or departure event on this path.

        Input:
            link        the event of a photon's arrival or departure.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the path and arriving at the event.
                        +1 to return later events, corresponding to photons
                           arriving at the path after departing from the event.

            derivs      True to propagate derivatives of the link position into
                        the returned event. The time derivative is always
                        retained.

            guess       an initial guess to use as the event time along the
                        path; otherwise None. Should only be used if the event
                        time was already returned from a similar calculation.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         a tuple of two Events (path_event, link_event).

            path_event  the event on the path that matches the light travel time
                        from the link event. This event always has position
                        (0,0,0) on the path.

            link_event  a copy of the given event, with the photon arrival or
                        departure line of sight and light travel time filled in.

            If sign is +1, then these subfields and derivatives are defined.
                In path_event:
                    arr         direction of the arriving photon at the path.
                    arr_lt      (negative) light travel time from the link
                                event.
                In link_event:
                    dep         direction of the departing photon.
                    dep_lt      light travel time to the path_event.

            If sign is -1, then the new subfields are swapped between the two
            events. Note that subfield 'arr_lt' is always negative and 'dep_lt'
            is always positive. Subfields 'arr' and 'dep' always have the same
            direction in both events.

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

        # Handle derivatives
        if not derivs:
            link = link.without_derivs()        # preserves time-dependence

        # If the path has a shape of its own, QuickPaths are disallowed
        if self.shape != (): quick = None

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
        if type(quick) == dict:
            quick = quick.copy()
            quick['path_time_extension'] = limit
            quick['frame_time_extension'] = limit

        # Iterate to a solution for the light travel time "lt". Define
        #   y = separation_distance(time + lt) - sign * c * lt
        # where lt is negative for earlier linking events and positive for later
        # linking events.
        #
        # Solve for the value of lt at which y = 0, using Newton's method.
        #
        # Approximate the function as linear around the solution:
        #   y[n+1] - y[n] = (lt[n+1] - lt[n]) * dy_dlt
        # Our goal is for the next value of y, y[n+1], to equal zero. Our most
        # recent guess is (lt[n], y[n]).
        #
        # What should we use for lt[n+1]?
        #   lt[n+1] = lt[n] - y[n] / dy_dlt
        #
        # The function y is shown above. Its derivative is
        #   dy_dlt = outward_speed - sign * c

        # Interpret the sign
        signed_c = sign * constants.C
        if sign < 0.:
            path_key = 'dep'
            link_key = 'arr'
        else:
            link_key = 'dep'
            path_key = 'arr'

        # If the link is entirely masked...
        if np.all(link.mask):
            path_event = Event(link.time.all_masked(True),
                               link.state.all_masked(True),
                               self.waypoint, self.frame.wayframe)
            path_event.insert_subfield(path_key, Vector3.MASKED)
            path_event.insert_subfield(path_key + '_lt', Scalar.MASKED)

            new_link = link.replace(link_key, Vector3.MASKED,
                                    link_key + '_lt', Scalar.MASKED)

            return (path_event, new_link)

        # Define the path and the link event relative to the SSB in J2000
        link_wod = link.without_derivs()
        link_time = link_wod.time
        link_wrt_ssb = link.wrt_ssb(derivs=True, quick=quick)
        link_pos_ssb = link_wrt_ssb.pos
        link_vel_ssb = link_wrt_ssb.vel

        # Make initial guesses at the path event time
        path_wrt_ssb = self.wrt(Path.SSB, Frame.J2000)

        if guess is not None:
            path_time = guess
            lt = path_time - link_time
        else:
            lt = (path_wrt_ssb.event_at_time(link_time, quick=quick).pos -
                  link_pos_ssb).norm() / signed_c
            path_time = link_time + lt

        # Set light travel time limits to avoid a diverging solution
        lt_min = (path_time - link_time).min() - limit
        lt_max = (path_time - link_time).max() + limit

        # Broadcast the path_time to encompass the shape of the path, if any
        shape = Qube.broadcasted_shape(path_time, link.shape)
        if path_time.shape != shape:
            path_time = path_time.broadcast_into_shape(shape)

        # Iterate a fixed number of times or until the threshold of error
        # tolerance is reached. Convergence takes just a few iterations.
        max_dlt = np.inf
        for iter in range(iters):

            # Quicken the path and frame evaluations (but no more than once)
            path_wrt_ssb = path_wrt_ssb.quick_path(path_time, quick=quick)

            # Evaluate the position photon's curren SSB position based on time
            path_event_ssb = path_wrt_ssb.event_at_time(path_time, quick=False)
            delta_pos_ssb = path_event_ssb.pos - link_pos_ssb
            delta_vel_ssb = path_event_ssb.vel - link_vel_ssb

            dlt = ((delta_pos_ssb.norm() - lt * signed_c) /
                   (delta_vel_ssb.proj(delta_pos_ssb).norm() - signed_c))
            new_lt = (lt - dlt).clip(lt_min, lt_max, False)
            dlt = lt - new_lt 
            lt = new_lt

            # Re-evaluate the path time
            path_time = link_time + lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations:
                print LOGGING.prefix, 'Path._solve_photon', iter, max_dlt

            if max_dlt <= precision or max_dlt >= prev_max_dlt or \
               max_dlt == Scalar.MASKED:
                    break

        #### END OF LOOP

        # Construct the returned event
        path_event_ssb = path_wrt_ssb.event_at_time(path_time, quick=False)
        link_event_ssb = link_wrt_ssb.clone()

        # Put the derivatives back as needed (at least the time derivative)
        path_pos_ssb = path_event_ssb.state
        link_pos_ssb = link_event_ssb.state

        # Fill in the key subfields
        if sign > 0:
            ray_vector_ssb = (path_pos_ssb - link_pos_ssb).as_readonly()
        else:
            ray_vector_ssb = (link_pos_ssb - path_pos_ssb).as_readonly()

        lt = ray_vector_ssb.norm(derivs) / signed_c

        path_event_ssb = path_event_ssb.replace(path_key, ray_vector_ssb,
                                                path_key + '_lt', -lt)

        # Transform the path event into its origin and frame
        path_event = path_event_ssb.from_ssb(self, self.frame,
                                             derivs=True, quick=quick)

        # Transform the light ray into the link's frame
        new_link = link.replace(link_key + '_j2000', ray_vector_ssb,
                                link_key + '_lt', lt)

        return (path_event, new_link)

    ############################################################################
    # Path Generators
    ############################################################################

    def wrt(self, origin, frame=None):
        """Construct a path pointing from an origin to this target in any frame.

        This is overridden by SpicePath, where it is easy to connect any to
        paths defined within the SPICE system.

        Input:
            origin      an origin Path object or its registered name.
            frame       a frame object or its registered ID. Default is to use
                        the frame of the origin's path.
        """

        # Convert a Waypoint to its registered version
        # Sorry for the ugly use of "self"!
        if isinstance(self, Waypoint):
            self = Path.WAYPOINT_REGISTRY[self.path_id]

        # Convert the origin to a path
        origin = Path.as_path(origin)

        # Determine the coordinate frame
        frame = Frame.as_wayframe(frame)
        if frame is None:
            frame = origin.frame

        # Use this path if possible
        if self.origin == origin.waypoint:
            if self.frame == frame:
                return self
            else:
                return RotatedPath(self, frame)

        if origin.origin == self.waypoint:
            newpath = ReversedPath(origin)
            if newpath.frame == frame:
                return newpath
            else:
                return RotatedPath(newpath, frame)

        # If the path already exists, just return it
        key = (self.waypoint, origin.waypoint, frame)
        if key in Path.PATH_CACHE:
            return Path.PATH_CACHE[key]

        # If everything matches but the frame, return a RotatedPath
        key = (self.waypoint, origin.waypoint)
        if key in Path.PATH_CACHE:
            newpath = Path.PATH_CACHE[key]
            return RotatedPath(newpath, frame)

        # Look up the primary definition of this path
        try:
            path = Path.as_primary_path(self)
        except KeyError:
            # On failure, link from the origin path
            newpath = LinkedPath(self, self.origin.wrt(origin))
            if newpath.frame == frame:
                return newpath
            else:
                return RotatedPath(newpath, frame)

        # Look up the primary definition of the origin path
        try:
            origin = Path.as_primary_path(origin)
        except KeyError:
            # On failure, link through the origin's origin
            newpath = RelativePath(path.wrt(origin.origin, frame), origin)
            if newpath.frame == frame:
                return newpath
            else:
                return RotatedPath(newpath, frame)

        # If the path is an ancestor of the origin, reverse the direction and
        # try again
        if path in origin.ancestry:
            newpath = origin.wrt(path, frame)
            return ReversedPath(newpath)

        # Otherwise, search from the parent path and then link
        newpath = path.ancestry[0].wrt(origin, frame)
        newpath = LinkedPath(path, newpath)
        if newpath.frame == frame:
            return newpath
        else:
            return RotatedPath(newpath, frame)

    ########################################

    def quick_path(self, time, quick={}):
        """A new QuickPath that approximates this path within given time limits.

        A QuickPath operates by sampling the given path and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed
        up performance when the same path must be evaluated many times, e.g.,
        for every pixel of an image.

        Input:
            time        a Scalar defining the set of times at which the frame is
                        to be evaluated. Alternatively, a tuple (minimum time,
                        maximum time, number of times)
            quick       if None or False, no QuickPath is created and self is
                        returned; if another dictionary, then the values
                        provided override the values in the default dictionary
                        QUICK.dictionary, and the merged dictionary is used.
        """

        OVERHEAD = 500      # Assume it takes the equivalent time of this many
                            # evaluations just to set up the QuickPath.
        SPEEDUP = 5.        # Assume that evaluations are this much faster once
                            # the QuickPath is set up.
        SAVINGS = 0.2       # Require at least a 20% savings in evaluation time.

        # Make sure a QuickPath is requested
        if type(quick) != dict: return self

        # These subclasses do not need QuickPaths
        if type(self) in (QuickPath, Waypoint, AliasPath): return self

        # Obtain the local QuickPath dictionary
        quickdict = QUICK.dictionary
        if len(quick) > 0:
            quickdict = quickdict.copy()
            quickdict.update(quick)

        if not quickdict['use_quickpaths']: return self

        # Determine the time interval
        if type(time) in (list,tuple):
            (tmin, tmax, count) = time
        else:
            time = Scalar.as_scalar(time)
            tmin = time.min()
            tmax = time.max()
            count = np.size(time.values)

        if tmin == Scalar.MASKED: return self

        # If QuickPaths already exist...
        if not hasattr(self, 'quickpaths'):
            self.quickpaths = []

        # If the whole time range is already covered, just return this one
        for quickpath in self.quickpaths:
            if tmin >= quickpath.t0 and tmax <= quickpath.t1:

                if LOGGING.quickpath_creation:
                    print LOGGING.prefix, 'Re-using QuickPath: ' + str(self),
                    print '(%.3f, %.3f)' % (tmin, tmax)

                return quickpath

        # See if the overhead makes more work justified
        if count < OVERHEAD: return self

        # Get dictionary parameters
        dt = quickdict['path_time_step']
        extension = quickdict['path_time_extension']
        extras = quickdict['path_extra_steps']

        # Extend the time domain
        tmin -= extension
        tmax += extension

        # See if a QuickPath can be efficiently extended
        for quickpath in self.quickpaths:

            # If there's no overlap, skip it
            if (quickpath.t0 > tmax + dt) or (quickpath.t1 < tmin - dt):
                continue

            # Otherwise, check the effort involved
            duration = (max(tmax, quickpath.t1) - min(tmin, quickpath.t0))
            steps = int(duration//dt) - quickpath.times.size

            effort_extending_quickpath = OVERHEAD + steps + count/SPEEDUP
            if count >= effort_extending_quickpath: 
                if LOGGING.quickpath_creation:
                    print LOGGING.prefix, 'Extending QuickPath: ' + str(self),
                    print '(%.3f, %.3f)' % (tmin, tmax)

                quickpath.extend((tmin,tmax))
                return quickpath

        # Evaluate the effort using a QuickPath compared to the effort without
        steps = int((tmax - tmin)//dt) + 2*extras
        effort_using_quickpath = OVERHEAD + steps + count/SPEEDUP
        if count < (1. + SAVINGS) * effort_using_quickpath: 
            return self

        if LOGGING.quickpath_creation:
            print LOGGING.prefix, 'New QuickPath: ' + str(self),
            print '(%.3f, %.3f)' % (tmin, tmax)

        result = QuickPath(self, (tmin, tmax), quickdict)

        if len(self.quickpaths) > quickdict['quickpath_cache']:
            self.quickpaths = [result] + self.quickpaths[:-1]
        else:
            self.quickpaths = [result] + self.quickpaths

        return result

################################################################################
# Required subclasses
################################################################################

class Waypoint(Path):
    """Waypoint is a Path subclass used internally to identify a path by its
    path ID in the registry. It has no practical use outside of the registry.

    When evaluated, it always places itself at the origin of its own path. A
    Waypoint cannot be registered by the user."""

    def __init__(self, path_id, frame=None, shape=()):
        """Constructor for a Waypoint.

        Input:
            path_id     the path ID to use for both the origin and destination.
            frame       the frame to use; None for J2000.
            shape       shape of the path.
        """

        # Required attributes
        self.waypoint = self
        self.origin   = self
        self.frame    = Frame.as_wayframe(frame) or Frame.J2000
        self.path_id  = path_id
        self.shape    = shape
        self.keys     = set()

    def event_at_time(self, time, quick=False):
        return Event(time, Vector3.ZERO, self.origin, self.frame)

    # Registration does nothing
    def register(self):
        return

    def __str__(self):
        return ('Waypoint(' + self.path_id + '*' + self.frame_id + ')')

################################################################################

class AliasPath(Path):
    """An AliasPath takes on the properties of the path and frame it is given.

    Used to create a quick, temporary path that returns events relative to a
    particular path and frame. An AliasPath cannot be registered.
    """

    def __init__(self, path, frame=None):
        """Constructor for an AliasPath.

        Input:
            path        a path or the ID of the path it should emulate.
            frame       the frame in which events will be described. By default,
                        the frame associated with the given path.
        """

        self.path = Path.as_path(path)

        if frame is None:
            frame = self.path.frame
            self.rotation = None
        else:
            frame = Frame.as_frame(frame)
            if frame.wayframe != self.path.frame:
                self.rotation = frame.wrt(self.path.frame)
            else:
                self.rotation = None

        # Required attributes
        self.waypoint = self.path.waypoint
        self.path_id  = self.path.path_id
        self.origin   = self.path.origin
        self.frame    = frame.wayframe
        self.shape    = Qube.broadcasted_shape(self.path.shape,
                                               self.frame.shape)
        self.keys     = set()

    def register(self):
        raise TypeError('an AliasPath cannot be registered')

    def event_at_time(self, time, quick={}):
        event = self.path.event_at_time(time, quick=quick)

        if self.rotation is not None:
            event = event.rotate_by_frame(self.rotation, derivs=False,
                                                         quick=quick)
        return event

################################################################################

class LinkedPath(Path):
    """A LinkedPath adds one path to its immediate ancestor.

    The new path returns positions and velocities as offsets from the origin of
    the parent and in the parent's frame.
    """

    def __init__(self, path, parent):
        """Constructor for a Linked Path.

        Input:
            path        a path, which must be defined relative to the given
                        parent.
            parent      a path to which the above will be linked.
        """

        self.path = path
        self.parent = parent

        assert self.path.origin == self.parent.waypoint

        if self.path.frame == self.parent.frame:
            self.rotation = None
        else:
            self.rotation = self.path.frame.wrt(self.parent.frame)

        # Required attributes
        self.waypoint = path.waypoint
        self.path_id  = path.path_id
        self.origin   = parent.origin
        self.frame    = parent.frame
        self.shape    = Qube.broadcasted_shape(path.shape, parent.shape)
        self.keys     = set()

        if path.is_registered() and parent.is_registered():
            self.register()     # save for later use

    def event_at_time(self, time, quick={}):
        event = self.path.event_at_time(time, quick=quick)

        if self.rotation is not None:
            event = event.unrotate_by_frame(self.rotation, derivs=False,
                                                           quick=quick)

        return self.parent.add_to_event(event, derivs=False, quick=quick)

################################################################################

class RelativePath(Path):
    """RelativePath defines the separation between paths with a common origin.

    The new path uses the coordinate frame of the origin path.
    """

    def __init__(self, path, origin):
        """Constructor for a RelativePath.

        Input:
            path        a Path object or ID defining the endpoint of the path
                        returned.
            origin      a Path object or ID defining the origin and frame of the
                        path returned.
        """

        self.path       = Path.as_path(path)
        self.new_origin = Path.as_path(origin)

        assert self.path.origin.waypoint == self.new_origin.origin.waypoint

        if self.path.frame == self.new_origin.frame:
            self.rotation = None
        else:
            self.rotation = path.frame.wrt(origin.frame)

        # Required attributes
        self.waypoint = path.waypoint
        self.path_id  = path.path_id
        self.origin   = self.new_origin.waypoint
        self.frame    = self.new_origin.frame
        self.shape    = Qube.broadcasted_shape(path.shape, origin.shape)
        self.keys     = set()

        if path.is_registered() and origin.is_registered():
            self.register()     # save for later use

    def event_at_time(self, time, quick={}):
        event = self.path.event_at_time(time, quick=quick)

        if self.rotation is not None:
            event = event.unrotate_by_frame(rotation, derivs=False, quick=quick)

        return self.new_origin.subtract_from_event(event, derivs=False,
                                                          quick=quick)

################################################################################

class ReversedPath(Path):
    """ReversedPath generates the reversed Events from that of a given Path."""

    def __init__(self, path):
        """Constructor for a ReversedPath.

        Input:
            path        the Path object to reverse, or its registered ID.
        """

        self.path = Path.as_path(path)

        # Required attributes
        self.waypoint = self.path.origin.waypoint
        self.path_id  = self.path.origin.path_id
        self.origin   = self.path.waypoint
        self.frame    = self.path.frame
        self.shape    = self.path.shape
        self.keys     = set()

        if path.is_registered():
            self.register()     # save for later use

    def event_at_time(self, time, quick={}):
        event = self.path.event_at_time(time, quick=quick)
        return Event(event.time, -event.state, self.origin, self.frame)

################################################################################

class RotatedPath(Path):
    """RotatedPath returns event objects rotated to another coordinate frame."""

    def __init__(self, path, frame):
        """Constructor for a RotatedPath.

        Input:
            path        the Path object to rotate, or else its registered ID.
            frame       the Frame object by which to rotate the path, or else
                        its registered ID.
        """

        self.path = Path.as_path(path)
        self.rotation = Frame.as_frame(frame).wrt(path.frame)

        # Required attributes
        self.waypoint = path.waypoint
        self.path_id  = path.path_id
        self.origin   = path.origin
        self.frame    = self.rotation.wayframe
        self.shape    = path.shape
        self.keys     = set()

        if path.is_registered():
            self.register()     # save for later use

    def event_at_time(self, time, quick={}):
        event = self.path.event_at_time(time, quick=quick)
        return event.rotate_by_frame(self.rotation, derivs=False, quick=quick)

################################################################################

class QuickPath(Path):
    """QuickPath returns positions and velocities by interpolating another path.
    """

    def __init__(self, path, interval, quickdict):
        """Constructor for a QuickPath.

        Input:
            path        the Path object that this Path will emulate.
            interval    a tuple containing the start time and end time of the
                        interpolation, in TDB seconds.
            quickdict   a dictionary containing all the needed QuickPath
                        parameters.
        """

        if path.shape != ():
            raise ValueError('shape of QuickPath must be ()')

        self.slowpath = path

        self.waypoint = path.waypoint
        self.path_id =  path.path_id
        self.origin   = path.origin
        self.frame    = path.frame
        self.shape    = ()
        self.keys     = set()

        self.t0 = interval[0]
        self.t1 = interval[1]
        self.dt = quickdict['path_time_step']

        self.extras = quickdict['path_extra_steps']
        self.times = np.arange(self.t0 - self.extras * self.dt,
                               self.t1 + self.extras * self.dt + self.dt,
                               self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        self.events = self.slowpath.event_at_time(self.times, quick=False)
        self._spline_setup()

        # Test the precision
        precision_self_check = quickdict['path_self_check']
        if precision_self_check is not None:
            t = self.times[:-1] + self.dt/2.        # Halfway points

            true_event = self.slowpath.event_at_time(t, quick=False)
            (pos, vel) = self._interpolate_pos_vel(t)

            dpos = (true_event.pos - pos).norm() / (true_event.pos).norm()
            dvel = (true_event.vel - vel).norm() / (true_event.vel).norm()
            error = max(np.max(dpos.vals), np.max(dvel.vals))
            if error > precision_self_check:
                raise ValueError('precision tolerance not achieved: ' +
                                 str(error) + ' > ' +
                                 str(precision_self_check))

    ####################################

    def event_at_time(self, time, quick=False):
        (pos, vel) = self._interpolate_pos_vel(time)
        return Event(time, (pos,vel), self.origin, self.frame)

    def __str__(self):
        return 'QuickPath(' + self.path_id + '*' + self.frame_id + ')'

    def register(self):
        raise TypeError('a QuickPath cannot be registered')

    ####################################

    def _spline_setup(self):
        KIND = 3
        self.pos_x = interp.InterpolatedUnivariateSpline(self.times,
                                             self.events.pos.vals[:,0], k=KIND)
        self.pos_y = interp.InterpolatedUnivariateSpline(self.times,
                                             self.events.pos.vals[:,1], k=KIND)
        self.pos_z = interp.InterpolatedUnivariateSpline(self.times,
                                             self.events.pos.vals[:,2], k=KIND)

        self.vel_x = interp.InterpolatedUnivariateSpline(self.times,
                                             self.events.vel.vals[:,0], k=KIND)
        self.vel_y = interp.InterpolatedUnivariateSpline(self.times,
                                             self.events.vel.vals[:,1], k=KIND)
        self.vel_z = interp.InterpolatedUnivariateSpline(self.times,
                                             self.events.vel.vals[:,2], k=KIND)

    def _interpolate_pos_vel(self, time, collapse_threshold=None):

        if collapse_threshold is None:
            collapse_threshold = \
                QUICK.dictionary['quickpath_linear_interpolation_threshold']

        tflat = Scalar.as_scalar(time).flatten()
        tflat_max = np.max(tflat.vals)
        tflat_min = np.min(tflat.vals)
        time_diff = tflat_max - tflat_min

        pos = np.empty(tflat.shape + (3,))
        vel = np.empty(tflat.shape + (3,))

        if time_diff < collapse_threshold:
            # If all time values are basically the same, we only need to do
            # linear interpolation.
            tflat_diff = tflat.vals - tflat_min
            tflat2 = Scalar([tflat_min, tflat_max])
            pos_x = self.pos_x(tflat2.vals)
            pos_y = self.pos_y(tflat2.vals)
            pos_z = self.pos_z(tflat2.vals)
            vel_x = self.vel_x(tflat2.vals)
            vel_y = self.vel_y(tflat2.vals)
            vel_z = self.vel_z(tflat2.vals)
        
            if time_diff == 0.:
                pos[...,0] = pos_x[0]
                pos[...,1] = pos_y[0]
                pos[...,2] = pos_z[0]
                vel[...,0] = vel_x[0]
                vel[...,1] = vel_x[0]
                vel[...,2] = vel_x[0]
            else:
                pos[...,0] = ((pos_x[1]-pos_x[0])/time_diff * tflat_diff +
                              pos_x[0])
                pos[...,1] = ((pos_y[1]-pos_y[0])/time_diff * tflat_diff +
                              pos_y[0])
                pos[...,2] = ((pos_z[1]-pos_z[0])/time_diff * tflat_diff +
                              pos_z[0])
                vel[...,0] = ((vel_x[1]-vel_x[0])/time_diff * tflat_diff +
                              vel_x[0])
                vel[...,1] = ((vel_y[1]-vel_y[0])/time_diff * tflat_diff +
                              vel_y[0])
                vel[...,2] = ((vel_z[1]-vel_z[0])/time_diff * tflat_diff +
                              vel_z[0])

        else:            
            # Evaluate the positions and velocities
            pos[...,0] = self.pos_x(tflat.vals)
            pos[...,1] = self.pos_y(tflat.vals)
            pos[...,2] = self.pos_z(tflat.vals)
    
            vel[...,0] = self.vel_x(tflat.vals)
            vel[...,1] = self.vel_y(tflat.vals)
            vel[...,2] = self.vel_z(tflat.vals)

        # Return the positions and velocities
        return (Vector3(pos, tflat.mask).reshape(time.shape),
                Vector3(vel, tflat.mask).reshape(time.shape))

    ####################################

    def extend(self, interval):
        """Modify this QuickPath to extend the time interval."""

        # If the interval fits inside already, we're done
        if interval[0] >= self.t0 and interval[1] <= self.t1: return

        # Extend the interval
        if interval[0] < self.t0:
            count0 = int((self.t0 - interval[0]) // self.dt) + 1 + self.extras
            new_t0 = self.t0 - count0 * self.dt
            times  = np.arange(count0) * self.dt + new_t0
            event0 = self.slowpath.event_at_time(times, quick=False)
        else:
            count0 = 0
            new_t0 = self.t0

        if interval[1] > self.t1:
            count1 = int((interval[1] - self.t1) // self.dt) + 1 + self.extras
            new_t1 = self.t1 + count1 * self.dt
            times  = np.arange(count1) * self.dt + self.t1 + self.dt
            event1 = self.slowpath.event_at_time(times, quick=False)
        else:
            count1 = 0
            new_t1 = self.t1

        # Allocate the new arrays
        old_size = self.times.size
        new_size = old_size + count0 + count1
        pos_values = np.empty((new_size,3))
        vel_values = np.empty((new_size,3))

        # Copy the new arrays
        if count0 > 0:
            pos_values[0:count0,:] = event0.pos.values
            vel_values[0:count0,:] = event0.vel.values

        pos_values[count0:count0+old_size,:] = self.events.pos.values
        vel_values[count0:count0+old_size,:] = self.events.vel.values

        if count1 > 0:
            pos_values[count0+old_size:,:] = event1.pos.values
            vel_values[count0+old_size:,:] = event1.vel.values

        # Generate the new events
        self.times = np.arange(new_size) * self.dt + new_t0
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        new_events = Event(Scalar(self.times), (pos_values, vel_values),
                           self.events.origin, self.events.frame)
        self.events = new_events

        # Update the splines
        self._spline_setup()

################################################################################
# Initialization at load time...
################################################################################

# Initialize Path.SSB
Path.SSB = Waypoint('SSB')
Path.SSB.ancestry = []
Path.SSB.wrt_ssb = Path.SSB

# Initialize the registry
Path.initialize_registry()

###############################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Path(unittest.TestCase):

    def runTest(self):

        # Re-import here to so modules all come from the oops tree
        from oops.path_.path import Path, LinkedPath, ReversedPath, \
                                          RelativePath, RotatedPath, QuickPath

        # More imports are here to avoid conflicts
        import os
        import cspice
        from oops.path_.spicepath import SpicePath
        from oops.path_.linearpath import LinearPath
        from oops.frame_.spiceframe import SpiceFrame
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
        
        Path.USE_QUICKPATHS = False

        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

        # Registry tests
        Path.reset_registry()
        Frame.reset_registry()

        self.assertEquals(Path.WAYPOINT_REGISTRY["SSB"], Path.SSB)

        # LinkedPath tests
        sun = SpicePath("SUN", "SSB")
        earth = SpicePath("EARTH", "SUN")

        moon = SpicePath("MOON", "EARTH")
        linked = LinkedPath(moon, earth)

        direct = SpicePath("MOON", "SUN")

        times = np.arange(-3.e8, 3.01e8, 0.5e7)

        direct_event = direct.event_at_time(times)
        linked_event = linked.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((linked_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((linked_event.vel - direct_event.vel).norm() <= eps).all())

        # RelativePath
        relative = RelativePath(linked, SpicePath("MARS", "SUN"))
        direct = SpicePath("MOON", "MARS")

        direct_event = direct.event_at_time(times)
        relative_event = relative.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((relative_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((relative_event.vel - direct_event.vel).norm() <= eps).all())

        # ReversedPath
        reversed = ReversedPath(relative)
        direct = SpicePath("MARS", "MOON")

        direct_event = direct.event_at_time(times)
        reversed_event = reversed.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((reversed_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((reversed_event.vel - direct_event.vel).norm() <= eps).all())

        # RotatedPath
        rotated = RotatedPath(reversed, SpiceFrame("B1950"))
        direct = SpicePath("MARS", "MOON", "B1950")

        direct_event = direct.event_at_time(times)
        rotated_event = rotated.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((rotated_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((rotated_event.vel - direct_event.vel).norm() <= eps).all())

        # QuickPath tests
        moon = SpicePath("MOON", "EARTH")
        quick = QuickPath(moon, (-5.,5.), QUICK.dictionary)

        # Perfect precision is impossible
        try:
            quick = QuickPath(moon, np.arange(0.,100.,0.0001),
                              dict(QUICK.dictionary, **{"path_self_check":0.}))
            self.assertTrue(False, "No ValueError raised for PRECISION = 0.")
        except ValueError: pass

        # Timing tests...
        test = np.zeros(3000000)
        # ignore = moon.event_at_time(test, quick=False)  # takes about 15 sec
        ignore = quick.event_at_time(test) # takes maybe 2 sec

        Path.reset_registry()
        Frame.reset_registry()

        ################################
        # Test unregistered paths
        ################################

        ssb = Path.as_waypoint('SSB')

        slider1 = LinearPath(([3,0,0],[0,3,0]), 0., ssb)
        self.assertTrue(slider1.path_id.startswith('TEMPORARY'))

        event = slider1.event_at_time(1.)
        self.assertEqual(event.pos, (3,3,0))
        self.assertEqual(event.vel, (0,3,0))

        slider2 = LinearPath(([-2,0,0],[0,0,-2]), 0., slider1)
        self.assertTrue(slider2.path_id.startswith('TEMPORARY'))

        event = slider2.event_at_time(1.)
        self.assertEqual(event.pos, (-2,0,-2))
        self.assertEqual(event.vel, (0,0,-2))

        slider3 = LinearPath(([-1,0,0],[0,-3,2]), 0., slider2)
        self.assertTrue(slider3.path_id.startswith('TEMPORARY'))

        event = slider3.event_at_time(1.)
        self.assertEqual(event.pos, (-1,-3,2))
        self.assertEqual(event.vel, ( 0,-3,2))

        # Link unregistered frame to registered frame
        static = slider3.wrt(ssb)

        event = static.event_at_time(1.)
        self.assertEqual(event.pos, (0,0,0))
        self.assertEqual(event.vel, (0,0,0))

        # Link registered frame to unregistered frame
        static = ssb.wrt(slider3)

        event = static.event_at_time(1.)
        self.assertEqual(event.pos, (0,0,0))
        self.assertEqual(event.vel, (0,0,0))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
