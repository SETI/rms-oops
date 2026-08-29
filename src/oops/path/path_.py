##########################################################################################
# oops/path/path_.py
##########################################################################################

import re

from polymath              import Qube, Vector3
from oops.cache            import Cache
from oops.config           import PICKLE_CONFIG
from oops.event            import Event
from oops.frame.frame_     import Frame, J2000Frame
from oops.frame.spiceframe import SpiceFrame
from oops.mutable          import Mutable


class Path(Mutable):
    """Path is an abstract class that can return an Event (time, position and velocity)
    given a time or a Scalar of times. The coordinates are specified in a particular
    Frame and relative to another path. The method `event_at_time` generates these Events.

    Upon construction, each Path has a "primary definition" relative to its specified,
    pre-existing origin Path and reference Frame. For example, a `KeplerPath` describes an
    orbit relative to the Path and Frame of a central planet.

    Once a Path is defined, you can calculate Events (defined by a time and state
    vector) relative to different Paths and transform those events between different Paths
    and Frames. The method `wrt` (for "with respect to") returns a Path object whose
    `event_at_time` method converts Events between different origins and Frames.

    For example, suppose `saturn` is a Path defining the center of Saturn, using Saturn's
    rotating Frame. A lightning flash on the surface of Saturn might be described as an
    Event relative to the center of Saturn and in Saturn's Frame. In addition, suppose
    `cassini_wac` is a Path defining the position of the Cassini spacecraft, using the
    Frame of its wide-angle camera. Then the Path defined by::

        saturn_wrt_wac = saturn.wrt(cassini_wac)

    can be used to express the location of the same lightning flash in an image taken by
    the Cassini camera. In short, Path objects make it easy to convert Events between
    different Paths and Frames.

    Furthermore, the Path methods `photon_to_event` and `photon_from_event` will perform
    these calculations allowing for the light travel time, position, and stellar
    aberration for when an Event relative to one Path object is observed relative to
    another.

    Every Path also has a `waypoint` property, which provides a unique identifier for that
    Path without regard to its origin or Frame. In the example above, `saturn` and
    `saturn_wrt_wac` will have the same `waypoint`, meaning that they define Events on the
    same path, albeit relative to different Paths and Frames. The waypoint can be used in
    almost any place where the Path itself can be used, so this would also have worked::

        saturn_wrt_wac = saturn.waypoint.wrt(cassini_wac.waypoint)

    Note that it is possible to construct multiple Path objects that have the exact same
    primary definition. The first Path to be constructed with a particular definition
    will have a waypoint that points to itself. Subsequent Path objects that employ the
    same definition will all share this waypoint. As a result, you can determine if two
    Path objects are functionally equivalent by comparing their waypoint attributes using
    the `is` operator.

    Optionally, a Path can be registered under a path ID, which is a string that can be
    used globally to refer to that Path. You can use the `as_path` method to convert a
    Path ID to its Path. In most situations, a Path ID can be used in place of a Path. For
    example, if `saturn` is registered under the name "SATURN" and `cassini_wac` is
    registered under the name "WAC", then these expressions would also work::

        saturn_wrt_wac = saturn.wrt('WAC')
        saturn_wrt_wac = Path.as_path('SATURN').wrt('WAC')

    In general, two Paths cannot be assigned the same ID. If you attempt to reuse an ID,
    that ID will have a new version number appended to make it unique. For example, if a
    Path named "SATURN" already exists, the new Path will actually have the ID "SATURN-2".
    However, if the new Path is functionally identical to the existing path called
    "SATURN", then its ID will also be "SATURN".

    Properties:
        * path_id (str or None): The optional ID string for this Path. Once registered,
          a Path can be referenced globally by its Path ID.
        * stripped_id (str or None): The Path ID with any numeric suffix stripped; None if
          this Path is not registered.
        * string_id (str): The Path ID if this Path is registered; otherwise, a unique
          string derived from its Python id().
        * is_registered (bool): True if this Path is registered.
        * origin (Path): The Path relative to which state vectors are defined.
        * frame (Frame): The Frame used by coordinates that are returned by this Path's
          `event_at_time` method.
        * primary (Path): The primary definition of this Path.
        * waypoint (Path): A Path object that uniquely identifies this path, irrespective
          of any particular origin and frame. Under most circumstances, this is the Path's
          primary definition.
        * shape (tuple): The shape of the Path object. This is the shape of the Event
          object returned by `event_at_time` when it is called with a single time value.
        * wrt_ssb (Path): This Path relative to the Solar System Barycenter and J2000.
    """

    _Body = None                # Filled in by oops/__init__.py
    _QuickPath = None           # Filled in by oops/path/quickpath.py

    # Override to True if the class uses QuickPaths. Set this True only if this Path's
    # own events are costly to evaluate and vary with time; a Path that returns a fixed
    # Event has nothing to tabulate, and one that is already cheap gains nothing.
    # LinkedPath, RelativePath, ReversedPath, and RotatedPath inherit the value from the
    # paths they combine.
    _USE_QUICKPATHS = False

    # Set False to disable the class-specific shortcuts returned by _get_shortcut(), which
    # is useful when debugging: the shortcut and the general ancestry walk must agree, so
    # a discrepancy that appears only with shortcuts enabled localizes the fault. This is
    # read from Path itself rather than from the instance, so that a subclass cannot
    # shadow it and leave part of the hierarchy still taking shortcuts.
    _USE_SHORTCUTS = True

    ######################################################################################
    # Serialization support
    ######################################################################################

    @property
    def pickle_quickpath_details(self):
        """True if the full tabulation of all QuickPaths is to be included when pickling
        this Path.
        """
        if not hasattr(self, '_pickle_quickpath_details'):
            return PICKLE_CONFIG.quickpath_details
        return self._pickle_quickpath_details

    @pickle_quickpath_details.setter
    def pickle_quickpath_details(self, value):
        """Set to True to include the internal tabulations of all QuickPaths when pickling
        this Path.
        """
        self._pickle_quickpath_details = bool(value)

    def _get_quickpaths(self):
        """The `_quickpaths` attribute if present and needed for pickling, else None."""
        if (self._pickle_quickpath_details and hasattr(self, '_quickpaths')
                and self._quickpaths):
            return self._quickpaths
        return None

    ######################################################################################
    # Each subclass must override...
    ######################################################################################

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.

        Raises:
            NotImplementedError: If the subclass does not override this method.
            ValueError: If the shapes of `time` and this object cannot be broadcasted.

        Notes:
            The time and this Path object are not required to have the same shape;
            standard rules of broadcasting apply.
        """

        raise NotImplementedError(f'{type(self).__name__}.event_at_time is not '
                                  'implemented')

    ######################################################################################
    # String operations
    ######################################################################################

    def __str__(self):
        path_id = self.string_id
        frame_id = self._frame.string_id

        if self._origin is Path.SSB:
            if self._frame is Frame.J2000:
                return (f'{type(self).__name__}({path_id})')
            else:
                return (f'{type(self).__name__}({path_id}/{frame_id})')
        else:
            origin_id = self._origin.string_id
            if self._frame is Frame.J2000:
                return (f'{type(self).__name__}({path_id}-{origin_id})')
            else:
                return (f'{type(self).__name__}([{path_id}-{origin_id}]/{frame_id})')

    def __repr__(self):
        return self.__str__()

    def show(self, level, indent=0):
        if (level == 0 and self._path_id and self._origin is Path.SSB
                and self._frame is Frame.J2000):
            return '"' + self._path_id + '"'

        if level <= 1:
            return str(self)

        return self._show(level, indent)

    @property
    def waypoint(self):
        """The canonical version of this Path, used as a global key for indexing."""
        return self._waypoint

    @property
    def primary(self):
        """The primary definition of this Path."""
        return self._primary

    @property
    def origin(self):
        """The origin Path relative to which this Path is defined."""
        return self._origin

    @property
    def frame(self):
        """The Frame relative to which this Path is defined."""
        return self._frame

    @property
    def shape(self):
        """The shape of this Path as a tuple of integers."""
        return self._shape

    @property
    def path_id(self):
        """The ID of this Path as a string if registered; otherwise, None."""
        if not hasattr(self, '_path_id'):   # occurs in errors during Path initialization
            self._path_id = None
        return self._path_id

    _PATH_ID_PATTERN = re.compile(r'(.*)-\d+$')

    @property
    def stripped_id(self):
        """The Path ID of this object with any numeric suffix stripped; None if there is
        no ID.
        """
        if not self._path_id:
            return None
        match = Path._PATH_ID_PATTERN.match(self._path_id)
        if match:
            return match.group(1)
        return self._path_id

    @property
    def string_id(self):
        """The ID of this Path if it is registered; otherwise, a unique string derived
        from its Python id().
        """
        return self._path_id if self._path_id else f'#{id(self)}'

    @property
    def wrt_ssb(self):
        """This Path with respect to the Solar System Barycenter and J2000."""
        if not hasattr(self, '_wrt_ssb') or self._wrt_ssb is None:
            self._wrt_ssb = self.wrt(Path.SSB)

        return self._wrt_ssb

    @property
    def is_registered(self):
        """True if this Path is registered."""
        return bool(self._path_id)

    ######################################################################################
    # Cache Management
    ######################################################################################

    _PATH_REGISTRY = {}     # path ID -> waypoint
    _PATH_CACHE = {}        # waypoint or (waypoint, origin) or (waypoint, origin, frame)
                            #       -> linked "wrt" path
    _PATH_SUBCLASSES = []   # list of all subclasses of Path

    @staticmethod
    def _reset_caches():
        """Reset the caches to their initial states. Mainly useful for debugging."""

        Path._PATH_REGISTRY.clear()
        Path._PATH_REGISTRY['SSB'] = Path.SSB
        Path._PATH_REGISTRY[None] = Path.SSB

        Path._PATH_CACHE.clear()
        Path._PATH_CACHE[Path.SSB] = Path.SSB
        Path._PATH_CACHE[Path.SSB, Path.SSB, Frame.J2000] = Path.SSB

        for subclass in Path._PATH_SUBCLASSES:
            if hasattr(subclass, '_WAYPOINTS'):
                subclass._WAYPOINTS.clear()

    def _register(self, path_id=None):
        """Fill in this Path's waypoint and path_id; register if necessary.

        Parameters:
            path_id (str, optional): Name under which to register this Path; omit to
                leave this Path un-registered.
        """

        # Fill in the _key and the waypoint. A class without a _WAYPOINTS dictionary
        # does not share waypoints; each of its instances is its own waypoint. Neither
        # does an object that is still unfrozen: its key is derived from parameters that
        # can still be fitted, so two Paths that agree now might not agree later.
        # _reregister() adds it to the pool once it has been frozen.
        if hasattr(type(self), '_WAYPOINTS'):
            self._key = Cache.clean_key(self._waypoint_key())
            if self._is_frozen():
                self._waypoint = self._WAYPOINTS.setdefault(self._key, self)
            else:
                self._waypoint = self
        else:
            self._waypoint = self

        # Fill in the path ID and register if necessary
        if path_id:

            # Make sure this ID doesn't already exist
            if path_id in Path._PATH_REGISTRY:

                # ...but it's OK if the ID matches that of its existing wayframe
                if self._waypoint != Path._PATH_REGISTRY[path_id]._waypoint:
                    # Otherwise, add a numeric suffix to make it unique
                    k = 2
                    while True:
                        alt_path_id = f'{path_id}-{k}'
                        if alt_path_id not in Path._PATH_REGISTRY:
                            break
                        k += 1
                    path_id = alt_path_id

            # Assign this ID to the path and, if necessary, the waypoint
            self._path_id = path_id
            if not self._waypoint._path_id:
                self._waypoint._path_id = path_id

            # Register under this unique ID
            Path._PATH_REGISTRY[path_id] = self

        else:
            self._path_id = None

        # Update the Path cache
        self._origin = self._origin._waypoint       # make sure it's a waypoint
        self._frame = self._frame._wayframe
        if self._waypoint in Path._PATH_CACHE:
            self._wrt_ssb = Path._PATH_CACHE[self._waypoint].wrt_ssb
        else:
            Path._PATH_CACHE[self._waypoint] = self
            Path._PATH_CACHE[self._waypoint, self._origin] = self
            Path._PATH_CACHE[self._waypoint, self._origin, self._frame] = self

            # Follow this path's ancestry and register a LinkedPath at each step
            path = self
            while True:
                origin = path._origin
                if origin is Path.SSB:
                    break
                path = LinkedPath(path, origin)     # this saves to the cache
            self._wrt_ssb = path

        self._primary = self

    def _reregister(self):
        """Update this Path's key in the cache now that it has been frozen.

        A Path that was left out of the pool by `_register` because it was unfrozen is
        added to the pool here.
        """

        # A Path that is its own waypoint has no key to update
        if not hasattr(type(self), '_WAYPOINTS'):
            return

        # Remove the definition from the cache under the old key
        if self._key in self._WAYPOINTS and self._WAYPOINTS[self._key] is self:
            del self._WAYPOINTS[self._key]

        # Add the definition to the cache under the new key
        self._key = Cache.clean_key(self._waypoint_key())
        self._waypoint = self._WAYPOINTS.setdefault(self._key, self)

    @staticmethod
    def as_path(path):
        """The Path object given a path or its registered ID.

        Parameters:
            path (Path or str): The Path or the Path's ID.

        Returns:
            Path: The Path, converted from the ID if `path` is a string.

        Raises:
            KeyError: If `path` is an ID that has not been registered.
        """

        if isinstance(path, Path):
            return path

        return Path._PATH_REGISTRY[path]

    @staticmethod
    def as_primary_path(path):
        """The primary definition of a Path object.

        Parameters:
            path (Path or str): The Path or the Path's ID.

        Returns:
            Path: The Path representing this Path's primary definition.

        Raises:
            KeyError: If `path` is an ID string that has not been registered.
        """

        if isinstance(path, Path):
            return path._primary

        return Path._PATH_REGISTRY[path]

    @staticmethod
    def as_waypoint(path):
        """The waypoint (canonical definition) of a Path.

        If multiple Path objects have identical definitions, this is the first Path that
        was assigned this definition.

        Parameters:
            path (Path or str): The Path or the Path's ID.

        Returns:
            Path: The canonical Path, converted from the ID if `path` is a string.

        Raises:
            KeyError: If `path` is an ID string that has not been registered.
        """

        if isinstance(path, Path):
            return path._waypoint

        return Path._PATH_REGISTRY[path]

    @staticmethod
    def path_id_exists(path_id):
        """True if the given path ID exists in the registry.

        Parameters:
            path_id (str): An ID string.

        Returns:
            bool: True if a Path has been registered under this ID.
        """

        return path_id in Path._PATH_REGISTRY

    ######################################################################################
    # Event operations
    ######################################################################################

    # These must be defined here and not in Event.py, because that would create a circular
    # dependency in the order that modules are loaded.

    def subtract_from_event(self, event, *, derivs=True, quick=None):
        """An equivalent Event, but with this path redefining the origin.

        Parameters:
            event (Event): The Event from which this path is to be subtracted. This Path's
                origin must coincide with the Event's origin and the two objects must use
                the same frame.
            derivs (bool, optional): True to include derivatives in the attributes of the
                returned Event. The specific derivatives included will depend on the Path
                subclass and those within the given Event.
            quick (dict or bool, optional): An optional dictionary of parameter values to
                use as overrides to the configured default QuickPath and QuickFrame
                parameters; use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The given `event` relative to this Path.

        Raises:
            ValueError: If the Event's origin does not match this Path's origin, or if the
                two objects do not use the same frame.
        """

        # Check for compatibility
        if self._origin._waypoint != event.origin._waypoint:
            raise ValueError('Events must have a common origin for path subtraction')

        if self._frame._wayframe != event.frame._wayframe:
            raise ValueError('Events must share a common frame for path subtraction')

        # Create the path event
        path_event = self.event_at_time(event.time, quick=quick)

        # Strip derivatives from the given event if necessary
        if not derivs:
            event = event.wod
            path_event = path_event.wod

        # Subtract
        return Event(event.time, event.state - path_event.state, self, self.frame,
                     **event.subfields)

    def add_to_event(self, event, *, derivs=True, quick=None):
        """An equivalent Event, using the origin of this Path as the origin.

        Parameters:
            event (Event): The Event object to which this path is to be added. The Event's
                origin must be this Path, and the two objects must use the same frame.
            derivs (bool, optional): True to include derivatives in the attributes of the
                returned Event. The specific derivatives included will depend on the Path
                subclass and those within the given Event. Time derivatives are always
                retained.
            quick (dict or bool, optional): An optional dictionary of parameter values to
                use as overrides to the configured default QuickPath and QuickFrame
                parameters; use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The given `event` relative to the origin of this Path.

        Raises:
            ValueError: If the Event's origin is not this Path, or if the two objects do
                not use the same frame.
        """

        # Check for compatibility
        if self._waypoint != event.origin._waypoint:
            raise ValueError("An Event's origin must match this path for path addition")

        if self._frame._wayframe != event.frame._wayframe:
            raise ValueError('Events must share a common frame for path addition')

        # Create the path event
        path_event = self.event_at_time(event.time, quick=quick)

        # Strip derivatives from the given event if necessary
        if not derivs:
            event = event.wod

        # Add
        return Event(event.time, event.state + path_event.state, self.origin, event.frame,
                     **event.subfields)

    ######################################################################################
    # Path Generators
    ######################################################################################

    def _wrt(self, origin, frame=None, *, use_shortcuts=False):
        """This Path relative to the specified origin and frame.

        This is the private version. The public version is `wrt` and does not have the
        `use_shortcuts` option.

        Parameters:
            origin (Path or str): The origin Path defined by a Path object or its
                registered ID. Event coordinates are returned relative to this origin.
            frame (Frame or str, optional): A Frame object or its registered ID. Event
                coordinates are returned in this Frame. The default is to use the Frame of
                the `origin`.
            use_shortcuts (bool, optional): False to prevent checking for a class-specific
                shortcut.

        Returns:
            Path: This Path relative to the specified origin and frame.

        Raises:
            KeyError: If `origin` or `frame` is an ID string that has not been registered.
        """

        waypoint = self._waypoint
        origin = Path.as_path(origin)
        frame = (frame and Frame.as_wayframe(frame)) or origin._frame
        origin = origin._waypoint

        # If this Path was already cached, return it
        key = (waypoint, origin, frame)
        if key in Path._PATH_CACHE:
            return Path._PATH_CACHE[key]

        # See if there's a shortcut
        if use_shortcuts and Path._USE_SHORTCUTS:
            shortcut = self._get_shortcut(origin, frame)
            if shortcut:
                Path._PATH_CACHE[key] = shortcut
                return shortcut

        # Check for a reversal. Without this, linking a Path to one of its own descendants
        # would recurse forever, because the loop below only ever walks this Path's
        # ancestry and the ancestry of a root such as the SSB never advances.
        reversed_key = (origin, waypoint)
        if reversed_key in Path._PATH_CACHE:
            new_path = ReversedPath(Path._PATH_CACHE[reversed_key])
            if new_path._frame == frame:
                return new_path
            return RotatedPath(new_path, frame)

        # If the Path and origin match, just rotate
        if waypoint == origin:
            if frame == origin._frame:
                return NullPath(waypoint, frame=frame)
            else:
                return RotatedPath(waypoint, frame=frame)

        # Find the linked Path using the J2000 frame
        key = (waypoint, origin._waypoint, Frame.J2000)
        if key in Path._PATH_CACHE:
            new_path = Path._PATH_CACHE[key]
        else:
            # Otherwise, link through this Path's origin
            origin_path = self._origin._wrt(origin, Frame.J2000,
                                            use_shortcuts=use_shortcuts)
            new_path = LinkedPath(self, origin_path)

# TBD
#         # Fix the frames
#         if origin._frame != Frame.J2000:
#             if (test_key := (origin._waypoint, origin._waypoint, Frame.J2000
#             new_path = LinkedPath(new_path, RotatedPath(origin, frame))
#
#         if self._frame != Frame.J2000:
#             new_path = RotatedPath(new_path, self._frame)
        # If the frame is correct, we're done
        if new_path._frame == frame:
            return new_path

        # Otherwise, rotate and return
#         return RotatedPath(new_path, frame, )
        return RotatedPath(new_path, frame)

    def wrt(self, origin, frame=None):
        """This Path relative to the specified origin and frame.

        Parameters:
            origin (Path or str): The origin Path defined by a Path object or its
                registered ID.
            frame (Frame or str, optional): A Frame object or its registered ID. The
                default is to use the Frame of the `origin`.

        Returns:
            Path: This Path relative to the specified origin and frame.

        Raises:
            KeyError: If `origin` or `frame` is an ID string that has not been registered.
        """

        return self._wrt(origin, frame=frame, use_shortcuts=True)

    def _get_shortcut(self, origin, frame):
        """A Path that directly transforms from the given origin and frame to this Path.

        For most Path subclasses, this returns None. SpicePath overrides this method
        because the SPICE toolkit can directly link any SpicePath to any other SpicePath,
        with no intermediate steps required.

        Parameters:
            origin (Path): The origin Path, which must be a valid waypoint.
            frame (Frame): The Frame, which must be a valid wayframe.

        Returns:
            Path or None: The "shortcut" Path if it could be constructed.
        """

        return None

    def quick_path(self, time, quick=None):
        """A QuickPath that approximates this Path for the given range of times.

        A QuickPath operates by sampling the given Path and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed up
        performance when the same Path must be evaluated many times, e.g., for every pixel
        of an image.

        Parameters:
            time (Scalar): The time(s) at which this path is to be evaluated.
            quick (dict or bool, optional): If False, no QuickPath is created and `self`
                is returned; if a dictionary, then the values provided override the values
                in the default dictionary QUICK.dictionary, and the merged dictionary is
                used.

        Returns:
            QuickPath: A Path that approximates this Path for the given range of times,
            but can be evaluated quickly via splines.

        Notes:
            Any QuickPaths generated by this function are saved as a list inside
            `self._quickpaths`. If a pre-existing QuickPath that covers the time range is
            found in this list, it is returned rather than constructing a new QuickPath.
            If a QuickPath is found in the list that partially covers the time range, that
            QuickPath is extended to cover the full range and returned.
        """

        return Path._QuickPath.for_path(self, time, quick=quick)

##########################################################################################
# Utility Subclasses
##########################################################################################

class NullPath(Path):
    """A Path subclass that defines a Path relative to itself."""

    def __init__(self, path, frame=None):
        """Constructor for a NullPath.

        Parameters:
            path (Path or str): The waypoint or Path ID to use as the origin of the
                returned Path, from which it will always have a zero-valued offset.
            frame (Frame or str, optional): The wayframe or Frame ID to use in the
                returned Path; by default, this is the Path's `frame` attribute.

        Raises:
            KeyError: If `path` is an ID string that has not been registered.
        """

        path = Path.as_path(path)
        frame = frame and Frame.as_frame(frame) or path._frame

        self._waypoint = path._waypoint
        self._origin   = path._waypoint
        self._frame    = frame
        self._shape    = path._shape
        self._path_id  = path._path_id
        self._wrt_ssb  = path._wrt_ssb
        self._primary  = self

        key = (self._waypoint, self._waypoint, self._frame)
        if key not in Path._PATH_CACHE:
            Path._PATH_CACHE[key] = self

    def __str__(self):
        return f'NullPath({self.string_id})'

    def __getstate__(self):
        return (self._waypoint,)

    def __setstate__(self, state):
        (path,) = state
        self.__init__(path)

    def event_at_time(self, time, *, quick=False):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.
        """

        return Event(time, (Vector3.ZERO, Vector3.ZERO), self._origin, self._frame)


# This must be a singleton!
class SSBPath(NullPath):
    """The class for the Solar System Barycenter Path, relative to which all other Paths
    are defined.

    This class must be defined as a singleton.
    """

    _SSB = None
    _IS_IMMUTABLE = True

    def __new__(cls):
        if SSBPath._SSB is None:
            obj = super().__new__(cls)

            obj._waypoint = obj
            obj._origin   = obj
            obj._frame    = Frame.J2000
            obj._shape    = ()
            obj._path_id  = 'SSB'
            obj._primary  = obj
            obj._wrt_ssb  = obj

            # Emulate a SpicePath
            obj._spice_origin_code = 0
            obj._spice_origin_name = 'SSB'

            Path._PATH_REGISTRY['SSB'] = obj
            Path._PATH_REGISTRY['SOLAR SYSTEM BARYCENTER'] = obj
            Path._PATH_REGISTRY[None] = obj
            Path._PATH_CACHE[obj] = obj
            Path._PATH_CACHE[obj, obj, Frame.J2000] = obj

            SSBPath._SSB = obj

        return SSBPath._SSB

    def __init__(self):
        pass

    def __str__(self):
        return 'SSB'

    def _show(self, level, indent=0):
        return '"SSB"'

    @property
    def string_id(self):
        return 'SSB'

    def _get_shortcut(self, origin, frame):
        """A Path that directly transforms from the given origin and frame to this Path.

        This is an override of the default method, needed because the SPICE Toolkit can
        handle the connections between the SSB and SpicePaths very efficiently.

        Parameters:
            origin (Path): The origin Path, which must be a valid waypoint.
            frame (Frame): The Frame, which must be a valid wayframe.

        Returns:
            Path or None: The "shortcut" Path if it could be constructed.
        """

        if (isinstance(origin, Frame._SpicePath)
                and isinstance(frame, (SpiceFrame, J2000Frame))):
            return Path._SpicePath.get('SSB', origin, frame)

        return None


class LinkedPath(Path):
    """A LinkedPath applies one Path to another.

    The new path returns events relative to the origin and frame of a second Path.
    """

    def __init__(self, path, parent):
        """Constructor for a LinkedPath.

        Parameters:
            path (Path or str): The Path or ID defining a Path, which must be defined
                relative to the origin and frame of the given `parent`.
            parent (Path or str): The Path or ID to which `path` will be linked.

        Raises:
            KeyError: If `path` or `parent` is an ID string that has not been registered.
            ValueError: If `path`'s origin does not match `parent` or if the object shapes
                cannot be broadcasted.
        """

        path = Path.as_path(path)
        parent = Path.as_path(parent)
        if path._origin != parent._waypoint:
            raise ValueError(f'LinkedPath path mismatch: {path._origin}, '
                             f'{parent._waypoint}')

        self._path = path
        self._parent = parent
        self._USE_QUICKPATHS = (path._USE_QUICKPATHS or parent._USE_QUICKPATHS)
        if path._frame == parent._frame:
            self._rotation = None
        else:
            self._rotation = parent._frame.wrt(path._frame)

        self._waypoint = path._waypoint
        # This Path uses the frame of the parent Path. event_at_time() rotates into that
        # frame before the addition, so this is the frame of every Event it returns.
        self._frame    = parent._frame
        self._origin   = parent._origin
        self._shape    = Qube.broadcasted_shape(self._path._shape, self._parent._shape)
        self._path_id  = path._path_id
        self._primary  = self
        self._wrt_ssb  = None

        for key in [(self._waypoint, self._origin),
                    (self._waypoint, self._origin, self._frame)]:
            if key not in Path._PATH_CACHE:
                Path._PATH_CACHE[key] = self

    def __getstate__(self):
        return (self._path, self._parent)

    def __setstate__(self, state):
        self.__init__(*state)

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.
        """

        event = self._path.event_at_time(time, quick=quick)

        if self._rotation:
            event = event.rotate_by_frame(self._rotation, quick=quick)

        return self._parent.add_to_event(event, quick=quick)


class RelativePath(Path):
    """RelativePath defines the separation between paths with a common origin.

    The new path uses the coordinate frame of the origin path.
    """

    def __init__(self, path, origin):
        """Constructor for a RelativePath.

        Parameters:
            path (Path or str): The Path or ID defining the Path whose position is to be
                measured relative to `origin`.
            origin (Path or str): The Path or ID defining the new origin. It must have the
                same origin as `path`.

        Raises:
            KeyError: If `path` or `origin` is an ID string that has not been registered.
            ValueError: If `path` and `origin` have different origins or if the object
                shapes cannot be broadcasted.
        """

        path = Path.as_path(path)
        origin = Path.as_path(origin)
        if path._origin != origin._origin:
            raise ValueError(f'RelativePath origin mismatch: {path._origin}, '
                             f'{origin._origin}')

        self._path = path
        # Retain the origin Path itself, not just its waypoint. The waypoint is the
        # canonical definition of the same body but is defined relative to a different
        # origin, so it cannot perform the subtraction in event_at_time().
        self._new_origin = origin
        self._USE_QUICKPATHS = (path._USE_QUICKPATHS or origin._USE_QUICKPATHS)
        self._origin = origin._waypoint
        if path._frame == origin._frame:
            self._rotation = None
        else:
            self._rotation = origin._frame.wrt(path._frame)

        self._waypoint = path._waypoint
        # This Path uses the frame of the origin Path, as stated above. event_at_time()
        # rotates into that frame before the subtraction, so this is the frame of every
        # Event it returns.
        self._frame    = origin._frame
        self._shape    = Qube.broadcasted_shape(path._shape, origin._shape)
        self._path_id  = path._path_id
        self._primary  = self
        self._wrt_ssb  = None

        for key in [(self._waypoint, self._origin),
                    (self._waypoint, self._origin, self._frame)]:
            if key not in Path._PATH_CACHE:
                Path._PATH_CACHE[key] = self

    def __getstate__(self):
        return (self._path, self._new_origin)

    def __setstate__(self, state):
        self.__init__(*state)

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.
        """

        event = self._path.event_at_time(time, quick=quick)

        if self._rotation:
            event = event.rotate_by_frame(self._rotation, quick=quick)

        return self._new_origin.subtract_from_event(event, quick=quick)


class ReversedPath(Path):
    """ReversedPath generates the reversed Events from that of a given Path.

    The frame is that of the original origin.
    """

    def __init__(self, path):
        """Constructor for a ReversedPath.

        Parameters:
            path (Path or str): The Path or ID defining the Path to reverse.

        Raises:
            KeyError: If `path` is an ID string that has not been registered.
        """

        path = Path.as_path(path)
        self._path = path
        self._USE_QUICKPATHS = path._USE_QUICKPATHS

        self._waypoint = path._origin
        self._origin   = path._waypoint
        self._frame    = path._origin._frame
        self._shape    = path._shape
        self._path_id  = path._origin._path_id
        self._primary  = self
        self._wrt_ssb  = None

        key = (self._waypoint, self._origin, self._frame)
        if key not in Path._PATH_CACHE:
            Path._PATH_CACHE[key] = self

    def __getstate__(self):
        return (self._path,)

    def __setstate__(self, state):
        self.__init__(*state)

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.
        """

        event = self._path.event_at_time(time, quick=quick)
        return Event(event.time, -event.state, self._origin, self._frame)


class RotatedPath(Path):
    """RotatedPath returns event objects rotated to another coordinate frame."""

    def __init__(self, path, frame):
        """Constructor for a RotatedPath.

        Parameters:
            path (Path or str): The Path or ID defining the Path to rotate.
            frame (Frame or str): The Frame or ID defining the Frame into which to rotate
                coordinates when given relative to `path`'s origin and frame.

        Raises:
            KeyError: If `path` or `frame` is an ID string that has not been registered.
            ValueError: If the object shapes cannot be broadcasted.
        """

        path = Path.as_path(path)

        # The Frame's center of rotation need not match the Path's origin. Expressing a
        # state in rotating axes uses (d/dt)_rot A = (d/dt)_inertial A - omega x A, which
        # holds for any vector A; only the relative state being rotated is involved, not
        # the position of the center of rotation.
        rotation = Frame.as_frame(frame).wrt(path._frame)

        self._path = path
        self._rotation = rotation
        self._USE_QUICKPATHS = path._USE_QUICKPATHS

        self._waypoint = path._waypoint
        self._origin   = path._origin
        self._frame    = rotation._wayframe
        self._shape    = Qube.broadcasted_shape(path._shape, rotation._shape)
        self._path_id  = path._path_id
        self._primary  = self
        self._wrt_ssb  = None

        for key in [(self._waypoint, self._origin),
                    (self._waypoint, self._origin, self._frame)]:
            if key not in Path._PATH_CACHE:
                Path._PATH_CACHE[key] = self

    def __getstate__(self):
        return (self._path, self._rotation)

    def __setstate__(self, state):
        self.__init__(*state)

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.
        """

        event = self._path.event_at_time(time, quick=quick)
        return event.rotate_by_frame(self._rotation, quick=quick)

##########################################################################################
# Initialization at load time...
##########################################################################################

# Initialize Path.SSB
Path.SSB = SSBPath()

from ._photon_solver import photon_to_event, photon_from_event, _solve_photon
Path.photon_to_event   = photon_to_event
Path.photon_from_event = photon_from_event
Path._solve_photon     = _solve_photon

##########################################################################################
