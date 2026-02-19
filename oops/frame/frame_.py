##########################################################################################
# oops/frame/frame_.py: Abstract class Frame and its required subclasses
##########################################################################################

import re

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.cache     import Cache
from oops.config    import PICKLE_CONFIG
from oops.transform import Transform


class Frame:
    """A Frame is an abstract class that can return a Transform (rotation matrix and
    optional spin vector) given a time or Scalar of times. This Transform converts from a
    specified "reference" frame to this frame's coordinates. The methods
    `transform_at_time` and `transform_at_time_if_possible` generate these Transforms.

    Upon construction, each Frame has a "primary definition" relative to its specified,
    pre-existing reference Frame. For example, a `SpinFrame` describes how to transform
    from its reference Frame to a new Frame that is spinning at a fixed rate relative to
    the specified origin. You might use a SpinFrame to define the rotation of a planet
    relative to that planet's center.

    Once a Frame is defined, you can calculate Transforms from any other Frame to this
    one. The method `wrt` (for "with respect to"), lets you specify any reference Frame
    and it returns a new Frame object whose `transform_at_time` method will return this
    alternative Transform. Internally, OOPS determines the sequence of steps that are
    required to connect any Frame to any other Frame.

    For example, suppose `enceladus_iau` is a SpinFrame defining the rotation of Enceladus
    relative to its center. In addition, suppose 'cassini_wac' is a Frame defining the
    orientation of the Cassini Wide Angle Camera. Then the Frame defined by::

        wac_wrt_enceladus = cassini_wac.wrt(enceladus_iau)

    will return, for any given time(s), the Transform from a vector fixed on the surface
    of Enceladus, and relative to the center of Enceladus, to a new vector defining a line
    of sight in the Cassini camera's field of view.

    Every Frame also has a `wayframe` property, which provides a unique identifier for
    that Frame without regard to its reference or definition. In the example above,
    `wac_wrt_enceladus` and `cassini_wac` will have the same `wayframe`, meaning that they
    transform to the same Frame. The wayframe can be used in almost any place where the
    Frame itself can be used, so this would also have worked::

        wac_wrt_enceladus = cassini_wac.wayframe.wrt(enceladus_iau.wayframe)

    Note that it is possible to construct multiple Frame objects that have the exact same
    primary definition. The first Frame to be constructed with a particular definition
    will have a wayframe that points to itself. Subsequent Frame objects that employ the
    same definition will all share this wayframe. As a result, you can determine if two
    Frame objects are functionally equivalent by comparing their wayframe attributes using
    the `is` operator.

    Optionally, a Frame can be registered under a Frame ID, which is a string that can be
    used globally to refer to that Frame. You can use the `as_frame` method to convert a
    Frame ID to a Frame. In most situations, a Frame ID can be used in place of a Frame.
    For example, if `enceladus_iau` is registered under the name "ENCELADUS" and
    `cassini_wac` is registered under the name "WAC", then these expressions would also
    work::

        wac_wrt_enceladus = cassini_wac.wrt('ENCELADUS')
        wac_wrt_enceladus = Frame.as_frame('WAC').wrt('ENCELADUS')

    In general, two Frames cannot be assigned the same ID. If you attempt to reuse an ID,
    that ID will have a new version number appended to make it unique. For example, if a
    Frame named "ENCELADUS" already exists, the new Frame will actually have the ID
    "ENCELADUS-2". However, if the new Frame is functionally identical to the existing
    frame called "ENCELADUS", then its ID will also be "ENCELADUS".

    Properties:
        * frame_id (str or None): The optional ID string for this Frame. Once registered,
          a Frame can be referenced globally by its Frame ID.
        * reference (Frame): The Frame from which this Frame transforms. The
          `transform_at_time` method will always return a Transform that converts from the
          reference Frame to this Frame.
        * primary (Frame): The primary definition of this Frame.
        * wayframe (Frame): A Frame object that uniquely identifies this frame,
          irrespective of any particular reference. Under most circumstances, this is the
          Frame's primary definition.
        * origin (Path or None): A Path object that uniquely identifies the origin
          relative to which this Frame is defined. For inertial Frames, this can be None.
        * shape (tuple): The shape of the Frame object. This is the shape of the Transform
          object returned by `transform_at_time` when it is called with a single time
          value.
    """

    _Event = None               # Filled in by oops/__init__.py
    _Path = None                # Filled in by oops/__init__.py
    _QuickFrame = None          # Filled in by oops/frame/quickframe.py
    _SpicePath = None           # Filled in by oops/path/spicepath.py

    _USE_QUICKFRAMES = False    # Override to True if the class uses QuickFrames

    ######################################################################################
    # Serialization support
    ######################################################################################

    @property
    def pickle_quickframe_details(self):
        """If True, the full tabulation of all QuickFrames is included when pickling this
        Frame.
        """
        if not hasattr(self, '_pickle_quickframe_details'):
            return PICKLE_CONFIG.quickframe_details
        return self._pickle_quickframe_details

    @pickle_quickframe_details.setter
    def pickle_quickframe_details(self, value):
        """True to include the internal tabulations of all QuickFrames when pickling this
        Frame.
        """
        self._pickle_quickframe_details = bool(value)

    def _get_quickframes(self):
        """The `_quickframes` attribute if present and needed for pickling, else None."""
        if (self.pickle_quickframe_details and hasattr(self, '_quickframes')
                and self._quickframes):
            return self._quickframes
        return None

    ######################################################################################
    # Each subclass must override...
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            The time and the Frame object are not required to have the same shape;
            standard rules of broadcasting apply.
        """

        raise NotImplementedError(f'{type(self).__name__}.transform_at_time is not '
                                  'implemented')

    def transform_at_time_if_possible(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method transform_at_time(), this variant tolerates times that raise cspyce
        errors. It returns a new time Scalar along with the new Transform, where both
        objects skip over the times at which the transform could not be evaluated.

        The default behavior is to assume that all times are valid. As a result, this
        function calls `transform_at_time`, but also returns the given time Scalar. This
        behavior is overridden by SpiceFrame, where occasional short gaps in a C-kernel
        can be tolerated as long as the QuickFrame can interpolate across them.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (tuple): The tuple (`newtimes`, `transform`), where:

            * `newtimes` (Scalar): Times at which `transform` has been provided; this may
              be a subset of the input times given because it omits times at which the
              Transform could not be evaluated.
            * `transform` (Transform): The Tranform applicable at `newtimmes`. It rotates
              vectors from the reference frame to this frame.
        """

        time = Scalar.as_scalar(time)
        return (time, self.transform_at_time(time, quick=quick))

    def node_at_time(self, time, *, quick=None):
        """Angle from the reference Frame's X-axis, along its X-Y plane, to the ascending
        node of this Frame's X-Y plane.

        Values always fall between 0 and 2*pi.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Scalar): At the specified times, the angle from the reference Frame's X-axis,
                along its X-Y plane, to the ascending node of this Frame's X-Y plane.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        frame = self.wrt(self._reference)
        xform = frame.transform_at_time(time, quick=quick)
        z_axis_wrt_j2000 = xform.unrotate(Vector3.ZAXIS)
        (x, y, _) = z_axis_wrt_j2000.to_scalars()
        return (y.arctan2(x) + Scalar.HALFPI) % Scalar.TWOPI

    ######################################################################################
    # String operations
    ######################################################################################

    def __str__(self):
        if self._reference is Frame.J2000:
            return f'{type(self).__name__}({self.string_id})'
        return f'{type(self).__name__}({self.string_id}/{self._reference.string_id})'

    def __repr__(self):
        return self.__str__()

    @property
    def wayframe(self):
        """The canonical version of this Frame, used as a global key for indexing."""
        return self._wayframe

    @property
    def primary(self):
        """The primary definition of this Frame."""
        return self._primary

    @property
    def reference(self):
        """The Frame relative to which this Frame is referenced."""
        return self._reference

    @property
    def origin(self):
        """The origin Path of this Frame if any; None if it is inertial."""
        return self._origin

    @property
    def is_inertial(self):
        """True if this frame is inertial."""
        if hasattr(self, '_is_inertial'):
            return self._is_inertial

        return self._origin is None

    @property
    def shape(self):
        """The shape of this Frame as a tuple of integers."""
        return self._shape

    @property
    def frame_id(self):
        """The ID of this Frame as a string if registered; otherwise, None."""
        if not hasattr(self, '_frame_id'):  # occurs in errors during Frame initialization
            self._frame_id = None
        return self._frame_id

    _FRAME_ID_PATTERN = re.compile(r'(.*)-\d+$')

    @property
    def stripped_id(self):
        """The frame ID of this object with any numeric suffix stripped; None if there is
        no ID.
        """
        if not self._frame_id:
            return None
        match = Frame._FRAME_ID_PATTERN.match(self._frame_id)
        if match:
            return match.group(1)
        return self._frame_id

    @property
    def string_id(self):
        """The ID of this Frame if it is registered; otherwise, a unique string derived
        from its Python id()."""
        return self._frame_id if self._frame_id else f'#{id(self)}'

    @property
    def wrt_j2000(self):
        """This Frame with respect to J2000."""
        if not hasattr(self, '_wrt_j2000') or self._wrt_j2000 is None:
            self._wrt_j2000 = self.wrt(Frame.J2000)

        return self._wrt_j2000

    @property
    def is_registered(self):
        """True if this Frame is registered."""
        return bool(self._frame_id)

    ######################################################################################
    # Cache Management
    ######################################################################################

    _FRAME_REGISTRY = {}    # frame ID -> wayframe
    _FRAME_CACHE = {}       # wayframe or (wayframe, reference) -> linked "wrt" frame
    _FRAME_SUBCLASSES = []  # list of all subclasses of Frame

    @staticmethod
    def _reset_caches():
        """Reset the caches to their initial states. Mainly useful for debugging."""

        Frame._FRAME_REGISTRY.clear()
        Frame._FRAME_REGISTRY['J2000'] = Frame.J2000
        Frame._FRAME_REGISTRY[None] = Frame.J2000

        Frame._FRAME_CACHE.clear()
        Frame._FRAME_CACHE[Frame.J2000] = Frame.J2000
        Frame._FRAME_CACHE[Frame.J2000, Frame.J2000] = Frame.J2000

        for subclass in Frame._FRAME_SUBCLASSES:
            if hasattr(subclass, '_WAYFRAMES'):
                subclass._WAYFRAMES.clear()

    def _register(self, frame_id=None):
        """Fill in this Frame's wayframe and frame_id; register if necessary.

        Parameters:
            frame_id (str, optional): Name under which to register this Frame; omit to
                leave this Frame un-registered.
        """

        # Fill in the _key and the wayframe
        if hasattr(type(self), '_WAYFRAMES'):
            self._key = Cache.clean_key(self._wayframe_key())
            self._wayframe = self._WAYFRAMES.setdefault(self._key, self)

        # Fill in the frame ID and register if necessary
        if frame_id:

            # Make sure this ID doesn't already exist
            if frame_id in Frame._FRAME_REGISTRY:

                # ...but it's OK if the ID matches that of its existing wayframe
                if self._wayframe != Frame._FRAME_REGISTRY[frame_id]._wayframe:
                    # Otherwise, add a numeric suffix to make it unique
                    k = 2
                    while True:
                        alt_frame_id = f'{frame_id}_{k}'
                        if alt_frame_id not in Frame._FRAME_REGISTRY:
                            break
                        k += 1
                    frame_id = alt_frame_id

            # Assign this ID to the frame and, if necessary, the wayframe
            self._frame_id = frame_id
            if not self._wayframe._frame_id:
                self._wayframe._frame_id = frame_id

            # Register under this unique ID
            Frame._FRAME_REGISTRY[frame_id] = self

        else:
            self._frame_id = None

        # Update the Frame cache
        self._reference = self._reference._wayframe     # make sure this is a wayframe
        if self._wayframe in Frame._FRAME_CACHE:
            self._wrt_j2000 = Frame._FRAME_CACHE[self._wayframe]._wrt_j2000
        else:
            Frame._FRAME_CACHE[self._wayframe] = self
            Frame._FRAME_CACHE[self._wayframe, self._reference] = self

            # Follow this frame's ancestry and cache a LinkedFrame at each step
            frame = self
            while True:
                reference = frame._reference
                if reference is Frame.J2000:
                    break
                frame = LinkedFrame(frame, reference)   # this saves to the cache
            self._wrt_j2000 = frame

        self._primary = self

    def _reregister(self):
        """Update this Frame's key in the cache if it has now been frozen."""

        # Remove the definition from the cache under the old key
        if self._key in self._WAYFRAMES and self._WAYFRAMES[self._key] is self:
            del self._WAYFRAMES[self._key]

        # Add the definition to the cache under the new key
        self._key = Cache.clean_key(self._wayframe_key())
        self._wayframe = self._WAYFRAMES.setdefault(self._key, self)

    @staticmethod
    def as_frame(frame):
        """The Frame given a Frame or its registered ID.

        Parameters:
            frame (Frame or str): The Frame or the Frame's ID string.

        Returns:
            (Frame): The Frame, converted from the ID if `frame` is a string.

        Raises:
            KeyError: If `frame` is an ID that has not been registered.
        """

        if isinstance(frame, Frame):
            return frame

        return Frame._FRAME_REGISTRY[frame]

    @staticmethod
    def as_primary_frame(frame):
        """The primary definition of a Frame object.

        Parameters:
            frame (Frame or str): The Frame or the Frame's ID string.

        Returns:
            (Frame): The Frame representing this Frame's primary definition.

        Raises:
            KeyError: If `frame` is an ID string that has not been registered.
        """

        if isinstance(frame, Frame):
            return frame._primary

        return Frame._FRAME_REGISTRY[frame]

    @staticmethod
    def as_wayframe(frame):
        """The wayframe (canonical definition) of a Frame.

        If multiple Frame objects have identical definitions, this is the first Frame that
        was assigned this definition.

        Parameters:
            frame (Frame or str): The Frame or the Frame's ID string.

        Returns:
            (Frame): The canonical Frame, converted from the ID if `frame` is a string.

        Raises:
            KeyError: If `frame` is an ID string that has not been registered.
        """

        if isinstance(frame, Frame):
            return frame._wayframe

        return Frame._FRAME_REGISTRY[frame]._wayframe

    @staticmethod
    def frame_id_exists(frame_id):
        """True if the given frame ID exists in the registry.

        Parameters:
            frame_id (str): An ID string.

        Returns:
            (bool): True if a Frame has been registered under this ID.
        """

        return frame_id in Frame._FRAME_REGISTRY

    ######################################################################################
    # Frame Generators
    ######################################################################################

    # Can be overridden by some classes such as SpiceFrame, where it is easier to make
    # connections between frames.
    def _wrt(self, reference, *, use_shortcuts=False):
        """A Frame that directly transforms from the given reference to this Frame.

        This is the private version. The public version is `wrt` and does not have the
        `use_shortcuts` option.

        Parameters:
            reference (Frame or str): The reference Frame defined by a Frame object or its
                registered ID.
            use_shortcuts (bool, optional): False to prevent checking for a class-specific
                shortcut.

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
        """

        wayframe = self._wayframe
        reference = Frame.as_frame(reference)

        # If this Frame was already cached, return it
        key = (wayframe, reference._wayframe)
        if key in Frame._FRAME_CACHE:
            return Frame._FRAME_CACHE[key]

        # See if there's a shortcut
        if use_shortcuts and hasattr(self, '_get_shortcut'):
            shortcut = self._get_shortcut(reference)
            if shortcut:
                Frame._FRAME_CACHE[key] = shortcut
                return shortcut

        # Check for a reversal
        reversed_key = (reference._wayframe, wayframe)
        if reversed_key in Frame._FRAME_CACHE:
            return ReversedFrame(Frame._FRAME_CACHE[reversed_key])

        # Check for a null transform
        if wayframe == reference:
            return NullFrame(frame)

        # Connect through this frame's reference, then link
        parent = self._reference._wrt(reference, use_shortcuts=use_shortcuts)
        return LinkedFrame(self, parent)

    def wrt(self, reference):
        """A Frame that directly transforms from the given reference to this Frame.

        Parameters:
            reference (Frame or str): The reference Frame defined by a Frame or Frame ID.

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
        """

        return self._wrt(reference, use_shortcuts=True)

    def _get_shortcut(self, reference):
        """A Frame that directly transforms from the given reference to this Frame.

        For most Frame subclasses, this returns None. SpiceFrame overrides this method
        because the SPICE toolkit can directly link any SpiceFrame to any other
        SpiceFrame, with no intermediate steps required.

        Parameters:
            reference (Frame): The reference Frame, which must be a valid wayframe.

        Returns:
            (Frame or None): The "shortcut" Frame if it could be constructed.
        """

        return None

    def quick_frame(self, time, *, quick=None):
        """A QuickFrame that approximates this Frame for the given range of times.

        A QuickFrame operates by sampling the given frame and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed up
        performance when the same frame must be evaluated repeatedly many times, e.g., for
        every pixel of an image.

        This function evaluates the set of times and only constructs a QuickFrame only if
        doing so is likely to speed up this and future evaluations. Otherwise, it returns
        this Frame.

        Parameters:
            time (Scalar or array-like): The times at which the frame is to be evaluated.
            quick (dict or bool, optional): If False, no QuickPath is created and `self`
                is returned; if a dictionary, then the values provided override the values
                in the default dictionary QUICK.dictionary, and the merged dictionary is
                used.

        Notes:
            Any QuickFrames generated by this function are saved as a list inside
            `self._quickframes`. If a pre-existing QuickFrame that covers the time range
            is found in this list, it is returned rather than constructing a new
            QuickFrame. If a QuickFrame is found in the list that partially covers the
            time range, that QuickFrame is extended to cover the full range and returned.
        """

        return Frame._QuickFrame.for_frame(self, time, quick=quick)

##########################################################################################
# Utility Subclasses
##########################################################################################

class NullFrame(Frame):
    """A frame subclass that transforms a frame to itself."""

    def __init__(self, frame):
        """Constructor for a NullFrame.

        Parameters:
            frame (Frame or str): The wayframe or frame ID to use as the returned Frame
                and its reference.

        Raises:
            KeyError: If `frame` is an ID string that has not been registered.
        """

        frame = Frame.as_frame(frame)
        self._wayframe    = frame._wayframe
        self._reference   = frame._wayframe
        self._origin      = frame._origin
        self._shape       = frame._shape
        self._wayframe    = frame._wayframe
        self._primary     = self
        self._frame_id    = frame._frame_id
        self._wrt_j2000   = frame._wrt_j2000
        self._is_inertial = frame.is_inertial

        key = (self._wayframe, self._wayframe)
        if key not in Frame._FRAME_CACHE:
            Frame._FRAME_CACHE[key] = self

    def __str__(self):
        return f'NullFrame({self.string_id})'

    def __getstate__(self):
        return (self._reference,)

    def __setstate__(self, state):
        (frame,) = state
        self.__init__(frame)

    def transform_at_time(self, time, *, quick=False):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            The time and this Frame object are not required to have the same shape;
            standard rules of broadcasting apply.
        """

        return Transform(Matrix3.IDENTITY, Vector3.ZERO, self._reference, self._reference,
                         self._origin)


# This must be a singleton!
class J2000Frame(NullFrame):
    """The class for the J2000 frame, relative to which all other Frames are defined.

    This class must be defined as a singleton.
    """

    _J2000 = None
    _IS_IMMUTABLE = True

    def __new__(cls):
        if J2000Frame._J2000 is None:
            obj = super().__new__(cls)

            obj._wayframe    = obj
            obj._reference   = obj
            obj._origin      = None
            obj._shape       = ()
            obj._frame_id    = 'J2000'
            obj._primary     = obj
            obj._wrt_j2000   = obj
            obj._is_inertial = True

            # Emulate a SpiceFrame
            obj._spice_frame_code = 1
            obj._spice_frame_name = 'J2000'

            Frame._FRAME_REGISTRY['J2000'] = obj
            Frame._FRAME_REGISTRY[None] = obj
            Frame._FRAME_CACHE[obj] = obj
            Frame._FRAME_CACHE[obj, obj] = obj

            J2000Frame._J2000 = obj

        return J2000Frame._J2000

    def __init__(self):
        pass

    def __str__(self):
        return 'J2000'

    @property
    def string_id(self):
        return 'J2000'

    def _get_shortcut(self, reference):
        """A SpiceFrame that directly transforms from the given reference to this Frame.

        This is an override of the default method, needed because the SPICE Toolkit can
        handle the connections between J2000 and SpiceFrames very efficiently.

        Parameters:
            reference (Frame): The reference Frame, which must be a valid wayframe.
        """

        if isinstance(reference, Frame._SpiceFrame):
            return Frame._SpiceFrame.get('J2000', reference)

        return None


class LinkedFrame(Frame):
    """A LinkedFrame applies one frame's transform to another.

    The new frame describes coordinates in one frame relative to the reference of the
    second Frame.
    """

    def __init__(self, frame, parent):
        """Constructor for a LinkedFrame.

        Parameters:
            frame (Frame): A frame, which must be defined relative to the given `parent`.
            parent (Frame): The frame to which the above will be linked.

        Raises:
            KeyError: If `frame` or `parent` is an ID string that has not been registered.
        """

        frame  = Frame.as_frame(frame)
        parent = Frame.as_frame(parent)
        if frame._reference != parent._wayframe:
            raise ValueError(f'LinkedFrame mismatch: {frame}, {parent._wayframe}')
        if parent._origin not in (frame._origin, Frame.J2000, None):
            raise ValueError(f'LinkedFrame origin mismatch: {frame._origin}, '
                             f'{parent._origin}')

        self._frame     = frame
        self._parent    = parent

        self._wayframe  = frame._wayframe
        self._reference = parent._reference
        self._origin    = frame._origin or parent._origin
        self._shape     = Qube.broadcasted_shape(frame._shape, parent._shape)
        self._frame_id  = frame._frame_id
        self._primary   = self
        self._wrt_j2000 = None

        key = (self._wayframe, self._reference)
        if key not in Frame._FRAME_CACHE:
            Frame._FRAME_CACHE[key] = self

    def __getstate__(self):
        return (self._frame, self._parent)

    def __setstate__(self, state):
        self.__init__(*state)

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            The time and the Frame object are not required to have the same shape;
            standard rules of broadcasting apply.
        """

        parent_xform = self._parent.transform_at_time(time, quick=quick)
        xform = self._frame.transform_at_time(time, quick=quick)
        return xform.rotate_transform(parent_xform)

    def transform_at_time_if_possible(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method transform_at_time(), this variant tolerates times that raise cspyce
        errors. It returns a new time Scalar along with the new Transform, where both
        objects skip over the times at which the transform could not be evaluated.

        The default behavior is to assume that all times are valid. As a result, this
        function calls transform_at_time, but also returns the given time Scalar. This
        behavior is overridden by SpiceFrame, where occasional short gaps in a C-kernel
        can be tolerated as long as a QuickFrame interpolates across them.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (tuple): The tuple (`newtimes`, `transform`), where:

            * `newtimes` (Scalar): Times at which `transform` has been provided; this may
              be a subset of the input times given.
            * `transform` (Transform): The Tranform applicable at `newtimmes`. It rotates
              vectors from the reference frame to this frame.
        """

        (time1, parent) = self.parent.transform_at_time_if_possible(time)
        (time2, xform) = self.frame.transform_at_time_if_possible(time1)
        if time1.shape != time2.shape:
            parent = self.parent.transform_at_time(time2)

        return (time2, xform.rotate_transform(parent))


class ReversedFrame(Frame):
    """A Frame that generates the inverse Transform of a given Frame."""

    def __init__(self, frame):
        """Constructor for a ReversedFrame.

        Parameters:
            frame (Frame): Frame to be reversed.

        Raises:
            KeyError: If `frame` is an ID string that has not been registered.
        """

        frame = Frame.as_frame(frame)
        self._frame     = frame
        self._reference = frame._wayframe
        self._origin    = frame._origin
        self._shape     = frame._shape
        self._wayframe  = frame._reference
        self._primary   = frame._reference
        self._frame_id  = frame._reference._frame_id

        key = (self._wayframe, self._reference)
        if key not in Frame._FRAME_CACHE:
            Frame._FRAME_CACHE[key] = self

    def __getstate__(self):
        return (self._frame,)

    def __setstate__(self, state):
        self.__init__(*state)

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            The time and the Frame object are not required to have the same shape;
            standard rules of broadcasting apply.
        """

        return self._frame.transform_at_time(time, quick=quick).invert()

    def transform_at_time_if_possible(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method transform_at_time(), this variant tolerates times that raise cspyce
        errors. It returns a new time Scalar along with the new Transform, where both
        objects skip over the times at which the transform could not be evaluated.

        The default behavior is to assume that all times are valid. As a result, this
        function calls transform_at_time, but also returns the given time Scalar. This
        behavior is overridden by SpiceFrame, where occasional short gaps in a C-kernel
        can be tolerated as long as a QuickFrame interpolates across them.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (tuple): The tuple (`newtimes`, `transform`), where:

            * `newtimes` (Scalar): Times at which `transform` has been provided; this may
              be a subset of the input times given.
            * `transform` (Transform): The Tranform applicable at `newtimmes`. It rotates
              vectors from the reference frame to this frame.
        """

        (time, xform) = self._frame.transform_at_time_if_possible(time, quick=quick)
        return (time, xform.invert())

##########################################################################################
# Initialization at load time...
##########################################################################################

# Initialize Frame.J2000
Frame.J2000 = J2000Frame()

##########################################################################################
