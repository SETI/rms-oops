################################################################################
# oops/path_/path_.py: Abstract class Path and its required subclasses
################################################################################

import numpy as np
import scipy.interpolate as interp
from polymath import *

from oops.config import QUICK, PATH_PHOTONS, LOGGING
from oops.event  import Event

import oops.constants as constants
import oops.registry  as registry

class Path(object):
    """A Path is an abstract class that returns an Event (time, position and
    velocity) given a Scalar time. The coordinates are specified in a particular
    frame and relative to another path. All paths are ultimately references to
    the Solar System Barycenter ("SSB") and the J2000 coordinate frame."""

########################################
# Each subclass must override...
########################################

    def __init__(self):
        """Constructor for a Path object. Every path must have these attributes:

        path_id         the ID of this Path, either a string or an integer.
        origin_id       the ID of the origin Path, relative to which this Path
                        is defined.
        frame_id        the ID of the frame used by the event objects returned.
        shape           the shape of this object, i.e, the shape of the event
                        returned when a single value of time is passed to
                        event_at_time().

        The primary definition of a path will be assigned these attributes by
        the registry:

        ancestry        a list of Path objects beginning with this one and
                        ending with with the Solar System Barycenter, where each
                        Path in sequence is the origin of the previous Path:
                            self.ancestry[0] = self.
                            self.ancestry[1] = origin path of self.
                            ...
                            self.ancestry[-1] = SSB in J2000.

        wrt_ssb         a definition of the same path relative to the Solar
                        System Barycenter, in the J2000 coordinate frame.
        """

        pass

    def event_at_time(self, time, quick=None):
        """Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.

        Return:         an event object containing (at least) the time, position
                        and velocity of the path.

        Note that the time and the path are not required to have the same shape;
        standard rules of broadcasting apply.
        """

        pass

    # string operations
    def __str__(self):
        return ("Path([" + self.path_id   + " - " +
                           self.origin_id + "]*" +
                           self.frame_id + ")")

    def __repr__(self): return self.__str__()

################################################################################
# Registry Management
################################################################################

    # The global registry is keyed in two ways. If the key is a string, then
    # this constitutes the primary definition of the path. The origin_id
    # of the path must already be in the registry, and the new path's ID
    # cannot be in the registry.
    
    # We also create secondary definitions of a path, where it is defined
    # relative to a different reference frame and/or with respect to a different
    # origin. These are entered into the registry keyed twice, by a tuple:
    #   (path_id, origin_id)
    # and a triple:
    #   (path_id, origin_id, frame_id).
    # This saves the effort of re-creating paths used repeatedly.

    @staticmethod
    def initialize_registry():
        """Initializes the registry. It is not generally necessary to call this
        function, but it can be used to reset the registry for purposes of
        debugging."""

        global WAYPOINT_SUBCLASS

        if registry.SSB is None:
            registry.SSB = Waypoint("SSB")
            registry.SSB.ancestry = [registry.SSB]
            registry.PATH_CLASS = Path

        registry.PATH_REGISTRY = {"SSB": registry.SSB,
                                  ("SSB","SSB"): registry.SSB,
                                  ("SSB","SSB","J2000"): registry.SSB}

    def register(self, shortcut=None):
        """Registers a Path definition. If the path's ID is new, it is assumed
        to be the primary definition and is keyed by the ID alone. However, a
        primary definition must use an origin ID that is already registered.

        Otherwise or in addition, two secondary keys are added to the registry
        if they are not already present:
            (path_id, reference_id)
            (path_id, reference_id, frame_id)
        These keys also point to the same Path object.

        If a shortcut name is given, then self is treated as a shortcut
        definition. The path is registered under the shortcut name and also
        under the triplet (path_id, reference_id, frame_id), but other
        registered definitions of the path are not modified.
        """

        # Make sure the registry is initialized
        if registry.PATH_REGISTRY == {}: Path.initialize_registry()

        # Handle a shortcut
        if shortcut is not None:
            registry.PATH_REGISTRY[shortcut] = self

            key = (self.path_id, self.origin_id, self.frame_id)
            registry.PATH_REGISTRY[key] = self
            return

        # Make sure the origin is registered
        origin = registry.PATH_REGISTRY[self.origin_id]

        # If the ID is unregistered, insert this as a primary definition
        if not registry.PATH_REGISTRY.has_key(self.path_id):
            registry.PATH_REGISTRY[self.path_id] = self

            # Fill in the ancestry too
            self.ancestry = [self] + origin.ancestry

            # Also define the "Waypoint" versions
            waypoint = Waypoint(self.path_id, self.frame_id)

            key = (self.path_id, self.path_id)
            registry.PATH_REGISTRY[key] = waypoint

            key = (self.path_id, self.path_id, self.frame_id)
            registry.PATH_REGISTRY[key] = waypoint

            # Also define the path with respect to the SSB
            if self.origin_id == "SSB" and self.frame_id == "J2000":
                self.wrt_ssb = self
            else:
                self.wrt_ssb = self.connect_to("SSB", "J2000")

                key = (self.path_id, "SSB", "J2000")
                registry.PATH_REGISTRY[key] = self.wrt_ssb

                key = (self.path_id, "SSB")
                if not registry.PATH_REGISTRY.has_key(key):
                    registry.PATH_REGISTRY[key] = self.wrt_ssb

        # If the tuple (self.frame_id, self.origin_id) is unregistered, insert
        # this as a secondary definition
        key = (self.path_id, self.origin_id)
        if not registry.PATH_REGISTRY.has_key(key):
            registry.PATH_REGISTRY[key] = self

        # If the triple (self.frame_id, self.origin_id, self.frame_id) is
        # unregistered, insert this as a tertiary definition
        key = (self.path_id, self.origin_id, self.frame_id)
        if not registry.PATH_REGISTRY.has_key(key):
            registry.PATH_REGISTRY[key] = self

    def unregister(self):
        """Removes this path from the registry."""

        # Note that we only delete the primary entry and any path in which this
        # is one of the end points. If the path is used as an intermediate step
        # between other paths, it will cease to be visible in the dictionary
        # but paths that use it will continue to function unchange.

        path_id = self.path_id
        for key in registry.PATH_REGISTRY.keys():
            if path_id == key: del registry.PATH_REGISTRY[key]

            if type(key) == type(()):
                if   path_id == key[0]: del registry.PATH_REGISTRY[key]
                elif path_id == key[1]: del registry.PATH_REGISTRY[key]

    def reregister(self):
        """Adds this frame to the registry, replacing any definition of the same
        name."""

        self.unregister()
        self.register()

    @staticmethod
    def lookup(key):
        return registry.PATH_REGISTRY[key]

################################################################################
# Event operations
################################################################################

# These must be defined here and not in Event.py, because that would create a
# circular dependency in the order that modules are loaded.

    def subtract_from_event(self, event, quick=None, derivs=False):
        """Returns an equivalent event, but with this path redefining its
        origin.

        Input:
            event       the event object from which this path is to be
                        subtracted. The path's origin must coincide with the
                        event's origin, and the two objects must use the same
                        frame.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.
            derivs      True to retain derivatives during this calculation;
                        false to remove them.
        """

        # Check for compatibility
        assert self.origin_id == event.origin_id
        assert self.frame_id  == event.frame_id

        # Create a new event by subtracting this path from the origin
        offset = self.event_at_time(event.time, quick)

        offset.pos.subarray_math = derivs
        offset.vel.subarray_math = derivs

        result = Event(event.time.copy(derivs),
                       event.pos - offset.pos,
                       event.vel - offset.vel,
                       self.path_id, event.frame_id)

        result.copy_subfields_from(event)
        return result

    def add_to_event(self, event, quick=None, derivs=False):
        """Returns an equivalent event, but with the origin of this path
        redefining its origin.

        Input:
            event       the event object to which this path is to be added.
                        The path's endpoint must coincide with the event's
                        origin, and the two objects must use the same frame.
            derivs      True to retain derivatives during this calculation;
                        false to remove them.
        """

        # Check for compatibility
        assert self.path_id  == event.origin_id
        assert self.frame_id == event.frame_id

        # Create a new event by subtracting this path from the origin
        offset = self.event_at_time(event.time, quick)

        offset.pos.subarray_math = derivs
        offset.vel.subarray_math = derivs

        result = Event(event.time.copy(derivs),
                       event.pos + offset.pos,
                       event.vel + offset.vel,
                       self.origin_id, event.frame_id)

        result.copy_subfields_from(event)
        return result

################################################################################
# Photon Solver
################################################################################

    def photon_to_event(self, link, quick=None, derivs=False, guess=None,
                              update=True,
                              iters=None, precision=None, limit=None):
        """Returns the departure event at the given path for a photon, given the
        linking event of the photon's arrival. See _solve_photon() for details.
        """

        return self._solve_photon(link, -1, quick, derivs, guess, update,
                                         iters, precision, limit)

    def photon_from_event(self, link, quick=None, derivs=False, guess=None,
                                update=True,
                                iters=None, precision=None, limit=None):
        """Returns the arrival event at the given path for a photon, given the
        linking event of the photon's departure. See _solve_photon() for details.
        """

        return self._solve_photon(link, +1, quick, derivs, guess, update,
                                         iters, precision, limit)

    def _solve_photon(self, link, sign, quick=None, derivs=False, guess=None,
                            update=True,
                            iters=None, precision=None, limit=None):

        """Solve for a photon event on this path, given that the other end of
        the photon's path is at another specified event (time and position).

        Input:

            link        the event of a photon's arrival or departure. The
                        returned event will be linked to this one.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the path and arriving at the event.
                        +1 to return later events, corresponding to photons
                           arriving at the path after departing from the event.

            quick       False to disable QuickPaths and QuickFrames; True for
                        the default options; a dictionary to override specific
                        options.

            derivs      True to include subfields containing the partial
                        derivatives with respect to the time of the linking
                        event

            guess       an initial guess to use as the event time along the
                        path; otherwise None. Should only be used if the event
                        time was already returned from a similar calculation.

            update      True to update the photon arrival or departure event in
                        the linking event; False to leave the linking event
                        unchanged.

            The following input parameters have default defined in file
            oops_.config.PATH_PHOTONS:

            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. Can be used to
                        limit divergence of the solution in some rare cases.

        Return:         the photon arrival event at the path, always given at
                        position (0,0,0) and velocity (0,0,0) relative to the
                        path. The following subfields are filled in for the
                        returned event and, if update is True, for the linking
                        event as well:

                        arr/dep         the vector from the departing photon
                                        event to the arriving photon event, in
                                        the frame of the path.

                        arr_lt/dep_lt   the light travel time separating the
                                        events. arr_lt is negative; dep_lt is
                                        positive.

                        If derivs is True, then the path event has these
                        subfields: time.d_dt, plus arr.d_dt or los.d_dt.
        """

        if self.shape != []: quick = False

        if iters is None:
            iters = PATH_PHOTONS.max_iterations
        if precision is None:
            precision = PATH_PHOTONS.dlt_precision
        if limit is None:
            limit = PATH_PHOTONS.dlt_limit

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
            path_key = "dep"
            link_key = "arr"
        else:
            link_key = "dep"
            path_key = "arr" 

        # If the link is entirely masked...
        if np.all(link.mask):
            return self._masked_link(link, sign, link_key, derivs)

        # Define the path and the linking event relative to the SSB in J2000
        link_wrt_ssb = link.wrt_ssb(quick, derivs=derivs)
        path_wrt_ssb = Path.connect(self, "SSB", "J2000")

        # Make initial guesses at the path event time
        if guess is not None:
            path_time = guess
            lt = path_time - link.time
        else:
            lt = (path_wrt_ssb.event_at_time(link.time, quick).pos
                                           - link_wrt_ssb.pos).norm() / signed_c
            path_time = link.time + lt

        # Set limits to avoid a diverging solution
        path_time_min = path_time.min() - limit
        path_time_max = path_time.max() + limit

        lt_min = (path_time_min - link.time).min()
        lt_max = (path_time_max - link.time).max()

        # Speed up the path and frame evaluations if requested
        # Interpret the quick parameters
        if quick is False:
            quickdict = False
        else:
            if type(quick) == type({}):
                quickdict = dict(QUICK.dictionary, **quick)
            else:
                quickdict = QUICK.dictionary
            quickdict = dict(quickdict, **{"path_extension": limit,
                                           "frame_extension": limit})

        path_wrt_ssb = path_wrt_ssb.quick_path(path_time, quick=quickdict)

        # Broadcast the path_time to encompass the shape of the path, if any
        shape = Array.broadcast_shape((path_time, link))
        if path_time.shape != shape:
            path_time = path_time.rebroadcast(shape).copy()

        # Iterate a fixed number of times or until the threshold of error
        # tolerance is reached. Convergence takes just a few iterations.
        max_dlt = np.inf
        for iter in range(iters):

            # Evaluate the position photon's curren SSB position based on time
            path_event_ssb = path_wrt_ssb.event_at_time(path_time)

            delta_pos_ssb = path_event_ssb.pos - link_wrt_ssb.pos
            delta_vel_ssb = path_event_ssb.vel - link_wrt_ssb.vel

            dlt = ((delta_pos_ssb.norm() - lt * signed_c) /
                   (delta_vel_ssb.proj(delta_pos_ssb).norm() - signed_c))
            new_lt = (lt - dlt).clip(lt_min, lt_max, False)
            dlt = lt - new_lt 
            lt = new_lt

            path_time = (link.time + lt).clip(path_time_min, path_time_max, False)

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations:
                print LOGGING.prefix, "Path._solve_photon", iter, max_dlt

            if type(max_dlt) == Scalar and np.all(max_dlt.mask):
                return self._masked_link(link, sign, link_key, derivs)

            if max_dlt <= precision or max_dlt >= prev_max_dlt: break

        # Update the path event one last time
        path_event_ssb = path_wrt_ssb.event_at_time(path_time)
        path_event_ssb.collapse_time()
        path_event_ssb.insert_subfield("link", link)
        path_event_ssb.insert_subfield("sign", sign)

        # Fill in the photon paths...

        # Photon path in SSB/J2000 frame
        delta_pos_ssb = path_event_ssb.pos - link_wrt_ssb.pos

        # If derivatives are needed, insert dlos/dt
        if derivs:

            # Determine the derivative of path event time WRT the linking time.
            delta_vel_ssb = path_event_ssb.vel - link_wrt_ssb.vel

            # Derive the small relativistic fix to the relative velocity
            dtime_dt = 1. + delta_vel_ssb.proj(delta_pos_ssb).norm() / signed_c
            path_event_ssb.time.insert_subfield("d_dt", dtime_dt)

            # Fill in the time derivative of los, here scaled to the full
            # distance between the two events
            dlos_dt = (path_event_ssb.vel * dtime_dt - link_wrt_ssb.vel)
            delta_pos_ssb.insert_subfield("d_dt", dlos_dt)

        # Update the photon info in the path event WRT SSB
        signed_los_ssb = sign * delta_pos_ssb
        path_event_ssb.insert_subfield(path_key, signed_los_ssb)
        path_event_ssb.insert_subfield(path_key + "_lt", -lt)

        # Update the linking event
        if update:

            link_wrt_ssb.insert_subfield(link_key, signed_los_ssb.plain())
            link_wrt_ssb.insert_subfield(link_key + "_lt", lt)

            link_frame = registry.connect_frames(link.frame_id, "J2000")
            link_transform = link_frame.transform_at_time(link.time, quick)

            signed_los_link = link_transform.rotate(signed_los_ssb,
                                                    derivs=derivs)
            # If derivs is True, then dlos/dt will be rotated and then combined
            # with any time-derivative arising from the frame rotation.

            link.insert_subfield(link_key, signed_los_link)
            link.insert_subfield(link_key + "_lt", lt)
            link.filled_ssb = link_wrt_ssb

        # Transform the path event back to its origin and frame
        path_event = path_event_ssb.wrt(self.path_id, self.frame_id, quick,
                                        derivs=derivs)
        path_event.filled_ssb = path_event_ssb
        return path_event

    def _masked_link(self, link, sign, link_key, derivs=False):
        """Returns an entirely masked path event."""

        path_event = link.masked_link(self.origin_id, self.frame_id, sign)

        if derivs:
            path_event.time.insert_subfield("d_dt", Scalar.all_masked())
            path_event.pos.insert_subfield( "d_dt",
                                                MatrixN.all_masked(item=[3,1]))

        link.insert_subfield(link_key, Vector3.all_masked())
        link.insert_subfield(link_key + "_lt", Scalar.all_masked())

        return path_event

################################################################################
# Path Generators
################################################################################

    @staticmethod
    def connect(target, origin, frame=None):
        """Returns a path that creates event objects in which vectors point
        from any origin path to any target path, using any coordinate frame.

        Input:
            target      the Path object or ID of the target path.
            origin      the Path object or ID of the origin path.
            frame       the Frame object of ID of the coordinate frame to use;
                        use None for the default frame of the origin.
        """

        # Convert to IDs
        target_id = registry.as_path_id(target)
        origin_id = registry.as_path_id(origin)

        if frame is None:
            frame_id = registry.as_path(origin).frame_id
        else:
            frame_id = registry.as_frame_id(frame)

        # If the path already exists, just return it
        try:
            key = (target_id, origin_id, frame_id)
            return Path.lookup(key)
        except KeyError: pass

        # If the path exists but the frame is wrong, return a rotated version
        try:
            key = (target_id, origin_id)
            newpath = Path.lookup(key)
            result = RotatedPath(newpath, frame_id)
            result.register()
            return result
        except KeyError: pass

        # Otherwise, construct it by other means...
        target_path = Path.lookup(target_id)
        return target_path.connect_to(origin_id, frame_id)

    # Can be overridden by some classes such as SpicePath, where it is easier
    # to make connections.
    def connect_to(self, origin, frame=None):
        """Returns a Path object in which events point from an arbitrary origin
        to this path, in an arbitrary frame. It is assumed that the desired path
        does not already exist in the registry. This is not checked, and need
        not be checked by any methods that override this one.

        Input:
            origin          an origin Path object or its registered name.
            frame           a frame object or its registered ID. Default is
                            to use the frame of the origin's path.
        """

        # Get the endpoint paths
        target = registry.as_primary_path(self)
        origin = registry.as_primary_path(origin)

        # Fill in the frame
        if frame is None:
            frame_id = origin.frame_id
        else:
            frame = registry.as_frame(frame)
            frame_id = frame.frame_id

        # If everything matches but the frame, return a RotatedPath
        if self.origin_id == origin.path_id:
            return RotatedPath(self, frame_id)

        # If the target is an ancestor of the origin, reverse the direction and
        # try again
        if target in origin.ancestry:
            path = Path.connect(origin, target, frame)
            return ReversedPath(path)

        # Otherwise, search from the parent path and then link
        path = Path.connect(target.ancestry[1], origin, frame)
        return LinkedPath(target, path)

    ############################################################################
    # 2/29/12: No longer needed but might still come in handy some day.
    ############################################################################

    @staticmethod
    def common_ancestry(path1, path2):
        """Returns a pair of ancestry lists for the two given paths, where both
        lists end at Paths with the same name."""

        # Identify the first common ancestor of both paths
        for i in range(len(path1.ancestry)):
            id1 = path1.ancestry[i].path_id

            for j in range(len(path2.ancestry)):
                id2 = path2.ancestry[j].path_id

                if id1 == id2:
                    return (path1.ancestry[:i+1], path2.ancestry[:j+1])

        return (path1.ancestry, path2.ancestry)     # should never happen

    @staticmethod
    def str_ancestry(tuple):
        """Creates a string presenting the contents of the tuple containing the
        common ancestry between two paths. For debugging only."""

        list = ["(["]

        for item in tuple:
            for path in item:
                list += [path.path_id,"\", \""]

            list.pop()
            list += ["\"], [\""]

        list.pop()
        list += ["\"])"]

        return "".join(list)

########################################

    def quick_path(self, time, quick):
        """Returns a new QuickPath object that provides accurate approximations
        to the position and velocity vectors returned by this path. It can speed
        up performance substantially when the same path must be evaluated
        repeatedly but within a narrow range of times.

        Input:
            time        a Scalar defining the set of times at which the frame is
                        to be evaluated.
            quick       if False, no QuickPath is created and self is returned;
                        if True, the default dictionary QUICK.dictionary is
                        used; if another dictionary, then the values provided
                        override the defaults and the merged dictionary is used.
        """

        OVERHEAD = 500      # Assume it takes the equivalent time of this many
                            # evaluations just to set up the QuickPath.
        SPEEDUP = 5.        # Assume that evaluations are this much faster once
                            # the QuickPath is set up.
        SAVINGS = 0.2       # Require at least a 20% savings in evaluation time.

        # Make sure a QuickPath is requested
        if not quick: return self

        # A Waypoint is too easy
        if type(self) == Waypoint: return self

        # A QuickPath would be redundant
        if type(self) == QuickPath: return self

        # Obtain the local QuickPath dictionary
        quickdict = QUICK.dictionary
        if type(quick) == type({}):
            quickdict = dict(quickdict, **quick)

        if not quickdict["use_quickpaths"]: return self

        # Determine the time interval and steps
        time = Scalar.as_scalar(time)
        vals = time.vals

        dt = quickdict["path_time_step"]
        extension = quickdict["path_time_extension"]
        extras = quickdict["path_extra_steps"]

        tmin = np.min(vals) - extension
        tmax = np.max(vals) + extension
        steps = (tmax - tmin)//dt + 2*extras

        # If QuickPaths already exists...
        if self.__dict__.has_key("quickpaths"):
            existing_quickpaths = self.quickpaths
        else:
            existing_quickpaths = []

        # If the whole time range is already covered, just return this one
        for quickpath in existing_quickpaths:
            if tmin >= quickpath.t0 and tmax <= quickpath.t1:

                if LOGGING.quickpath_creation:
                    print LOGGING.prefix, "Re-using QuickPath: " + str(self),
                    print (tmin, tmax)

                return quickpath

        # See if the overhead makes more work justified
        count = np.size(vals)
        if count < OVERHEAD: return self

        # See if a QuickPath can be efficiently extended
        for quickpath in existing_quickpaths:
            duration = (max(tmax, quickpath.t1) - min(tmin, quickpath.t0))
            steps = duration//dt - quickpath.times.size

            # Compare the effort involved in extending to the effort without
            effort_extending_quickpath = OVERHEAD + steps + count/SPEEDUP
            if count >= effort_extending_quickpath: 

                if LOGGING.quickpath_creation:
                    print LOGGING.prefix, "Extending QuickPath: " + str(self),
                    print (tmin, tmax)

                quickpath.extend((tmin,tmax))
                return quickpath

        # Evaluate the effort using a QuickPath compared to the effort without
        effort_using_quickpath = OVERHEAD + steps + count/SPEEDUP
        if count < (1. + SAVINGS) * effort_using_quickpath: 
            return self

        if LOGGING.quickpath_creation:
            print LOGGING.prefix, "New QuickPath: " + str(self), (tmin, tmax)

        result = QuickPath(self, (tmin, tmax), quickdict)

        if len(existing_quickpaths) > quickdict["quickpath_cache"]:
            self.quickpaths = [result] + existing_quickpaths[:-1]
        else:
            self.quickpaths = [result] + existing_quickpaths

        return result

################################################################################
# Define the required subclasses
################################################################################

class LinkedPath(Path):
    """LinkedPath is a Path subclass that returns the result of adding one path
    path to its immediate ancestor. The new path returns positions and
    velocities as offsets from the origin of the parent and in that parent's
    frame.
    """

    def __init__(self, path, parent):
        """Constructor for a Linked Path.

        Input:
            path        a path, which must be defined relative to the given
                        parent.
            parent      a path to which the above will be linked.
        """

        self.path = registry.as_path(path)
        self.parent = registry.as_path(parent)

        assert self.path.origin_id == self.parent.path_id

        if self.path.frame_id == self.parent.frame_id:
            self.frame = None
        else:
            self.frame = registry.connect_frames(self.path.frame_id,
                                                 self.parent.frame_id)

        # Required fields
        self.path_id   = self.path.path_id
        self.origin_id = self.parent.origin_id
        self.frame_id  = self.parent.frame_id
        self.shape     = Qube.broadcasted_shape(self.path, self.parent)

    def event_at_time(self, time, quick=None):
        event = self.path.event_at_time(time, quick)

        if self.frame is not None:
            event = event.unrotate_by_frame(self.frame, quick)

        event = self.parent.add_to_event(event, quick)
        return event

################################################################################

class RelativePath(Path):
    """RelativePath is a Path subclass that returns the relative separation
    between two paths that share a common origin. The new path uses the
    coordinate frame of the origin path.
    """

    def __init__(self, path, origin):
        """Constructor for a RelativePath.

        Input:
            path        a Path object or ID defining the endpoint of the path
                        returned.
            origin      a Path object or ID defining the origin and frame of the
                        path returned.
        """

        self.path   = registry.as_path(path)
        self.origin = registry.as_path(origin)

        assert self.path.origin_id == self.origin.origin_id

        # Required fields
        self.path_id   = self.path.path_id
        self.origin_id = self.origin.path_id
        self.frame_id  = self.origin.frame_id
        self.shape     = Qube.broadcasted_shape(self.path, self.origin)

        if self.path.frame_id == self.origin.frame_id:
            self.frame = None
        else:
            self.path_frame = registry.connect_frames(self.path.frame_id,
                                                      self.origin.frame_id)

    def event_at_time(self, time, quick=None):
        event = self.path.event_at_time(time, quick)

        if self.frame is not None:
            event = event.unrotate_by_frame(self.frame, quick)

        event = self.origin.subtract_from_event(event, quick)

        return event

################################################################################

class ReversedPath(Path):
    """ReversedPath is Path subclass that generates the reversed Events from
    that of a given Path."""

    def __init__(self, path):
        """Constructor for a ReversedPath.

        Input:
            path        the Path object to reverse, or its registered ID.
        """

        self.path = registry.as_path(path)

        # Required fields
        self.path_id   = self.path.origin_id
        self.origin_id = self.path.path_id
        self.frame_id  = self.path.frame_id
        self.shape     = self.path.shape

    def event_at_time(self, time, quick=None):
        event = self.path.event_at_time(time, quick)
        event.pos = -event.pos
        event.vel = -event.vel

        return event

################################################################################

class RotatedPath(Path):
    """RotatedPath is a Path subclass that returns event objects rotated to
    another coordinate frame."""

    def __init__(self, path, frame):
        """Constructor for a RotatedPath.

        Input:
            path        the Path object to rotate, or else its registered ID.
            frame       the Frame object by which to rotate the path, or else
                        its registered ID.
        """

        self.path = registry.as_path(path)
        self.frame_id = registry.as_frame_id(frame)
        self.frame = registry.connect_frames(self.frame_id, self.path.frame_id)

        # Required fields
        self.path_id   = self.path.path_id
        self.origin_id = self.path.origin_id
        self.shape     = self.path.shape

    def event_at_time(self, time, quick=None):
        event = self.path.event_at_time(time)
        event = event.rotate_by_frame(self.frame)

        return event

################################################################################

class Waypoint(Path):
    """Waypoint is a Path subclass that always returns the origin. It can be
    useful merely for turning a path ID into a Path object."""

    def __init__(self, path_id, frame_id="J2000"):
        """Constructor for a Waypoint.

        Input:
            path_id     the path ID to use for both the origin and destination."
            frame_id    the frame ID to use.
        """

        # Required fields
        self.path_id   = path_id
        self.origin_id = path_id
        self.frame_id  = frame_id
        self.shape     = []

    def event_at_time(self, time, quick=None):
        return Event.null_event(time, self.path_id, self.frame_id)

    def __str__(self):
        return "Waypoint(" + self.path_id + "*" + self.frame_id + ")"

################################################################################

class QuickPath(Path):
    """QuickPath is a Path subclass that returns positions and velocities based
    on interpolation of another path within a specified time window."""

    def __init__(self, path, interval, quickdict):
        """Constructor for a QuickPath.

        Input:
            path            the Path object that this Path will emulate.
            interval        a tuple containing the start time and end time of
                            the interpolation, in TDB seconds.
            quickdict       a dictionary containing all the QuickPath
                            parameters.
        """

        self.path = path
        self.path_id   = path.path_id
        self.origin_id = path.origin_id
        self.frame_id  = path.frame_id

        assert path.shape == []
        self.shape = []

        self.t0 = interval[0]
        self.t1 = interval[1]
        self.dt = quickdict["path_time_step"]

        self.extras = quickdict["path_extra_steps"]
        self.times = np.arange(self.t0 - self.extras * self.dt,
                               self.t1 + self.extras * self.dt + self.dt,
                               self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        self.events = self.path.event_at_time(self.times, quick=False)
        self._spline_setup()

        # Test the precision
        precision_self_check = quickdict["path_self_check"]
        if precision_self_check is not None:
            t = self.times[:-1] + self.dt/2.        # Halfway points

            true_event = self.path.event_at_time(t)
            (pos, vel) = self._interpolate_pos_vel(t)

            dpos = (true_event.pos - pos).norm() / (true_event.pos).norm()
            dvel = (true_event.vel - vel).norm() / (true_event.vel).norm()
            error = max(np.max(dpos.vals), np.max(dvel.vals))
            if error > precision_self_check:
                raise ValueError("precision tolerance not achieved: " +
                                  str(error) + " > " +
                                  str(precision_self_check))

    ####################################

    def event_at_time(self, time, quick=None):
        (pos, vel) = self._interpolate_pos_vel(time)
        return Event(time, pos, vel, self.origin_id, self.frame_id)

    def __str__(self):
        return "QuickPath(" + self.path_id + "*" + self.frame_id + ")"

    ####################################

    def _spline_setup(self):
        KIND = 3
        self.pos_x = interp.UnivariateSpline(self.times,
                                             self.events.pos.vals[:,0], k=KIND)
        self.pos_y = interp.UnivariateSpline(self.times,
                                             self.events.pos.vals[:,1], k=KIND)
        self.pos_z = interp.UnivariateSpline(self.times,
                                             self.events.pos.vals[:,2], k=KIND)

        self.vel_x = interp.UnivariateSpline(self.times,
                                             self.events.vel.vals[:,0], k=KIND)
        self.vel_y = interp.UnivariateSpline(self.times,
                                             self.events.vel.vals[:,1], k=KIND)
        self.vel_z = interp.UnivariateSpline(self.times,
                                             self.events.vel.vals[:,2], k=KIND)

    def _interpolate_pos_vel(self, time):

        # time can only be a 1-D array in the splines
        tflat = Scalar.as_scalar(time).flatten()
        pos = np.empty(tflat.shape + (3,))
        vel = np.empty(tflat.shape + (3,))

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
        """Modifies the given QuickPath if necessary to accommodate the given
        time interval."""

        # If the interval fits inside already, we're done
        if interval[0] >= self.t0 and interval[1] <= self.t1: return

        # Extend the interval
        if interval[0] < self.t0:
            count0 = int((self.t0 - interval[0]) // self.dt) + 1 + self.extras
            new_t0 = self.t0 - count0 * self.dt
            times  = np.arange(count0) * self.dt + new_t0
            event0 = self.path.event_at_time(times, quick=False)
        else:
            count0 = 0
            new_t0 = self.t0

        if interval[1] > self.t1:
            count1 = int((interval[1] - self.t1) // self.dt) + 1 + self.extras
            new_t1 = self.t1 + count1 * self.dt
            times  = np.arange(count1) * self.dt + self.t1 + self.dt
            event1 = self.path.event_at_time(times, quick=False)
        else:
            count1 = 0
            new_t1 = self.t1

        # Allocate the new arrays
        old_size = self.times.size
        new_size = old_size + count0 + count1
        pos_vals = np.empty((new_size,3))
        vel_vals = np.empty((new_size,3))

        # Copy the new arrays
        if count0 > 0:
            pos_vals[0:count0,:] = event0.pos.vals
            vel_vals[0:count0,:] = event0.vel.vals

        pos_vals[count0:count0+old_size,:] = self.events.pos.vals
        vel_vals[count0:count0+old_size,:] = self.events.vel.vals

        if count1 > 0:
            pos_vals[count0+old_size:,:] = event1.pos.vals
            vel_vals[count0+old_size:,:] = event1.vel.vals

        # Generate the new events
        self.times = np.arange(new_t0, new_t1 + self.dt/2., self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        new_events = Event(Scalar(self.times),
                           Vector3(pos_vals), Vector3(vel_vals),
                           self.events.origin_id, self.events.frame_id)
        self.events = new_events

        # Update the splines
        self._spline_setup()

################################################################################
# Initialize the registry
################################################################################

Path.initialize_registry()

###############################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Path(unittest.TestCase):

    def runTest(self):

        # NOTE: THIS UNIT TEST IS KNOWN TO NOT WORK STANDALONE BECAUSE THE
        # PATH CLASS IN THIS FILE ENDS UP BEING A DIFFERENT CLASS FROM
        # oops.PATH_.PATH.PATH. EVERYTHING IS FINE AS LONG AS YOU RUN FROM
        # ONE OF THE HIGHER-LEVEL UNITTESTER PROGRAMS.
        
        Path.USE_QUICKPATHS = False

        # Imports are here to avoid conflicts
        from oops.path_.spicepath import SpicePath
        from oops.frame_.spiceframe import SpiceFrame
        import os
        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY
        import cspice
        
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

        registry.SSB = None
        Path.initialize_registry()
        
        # Registry tests
        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        self.assertEquals(registry.PATH_REGISTRY["SSB"], registry.SSB)
        self.assertEquals(registry.PATH_CLASS, Path)
        
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

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################