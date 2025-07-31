################################################################################
# oops/frame/frame_.py: Abstract class Frame and its required subclasses
################################################################################

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from polymath       import Matrix, Matrix3, Quaternion, Qube, Scalar, Vector3
from oops.config    import QUICK, LOGGING, PICKLE_CONFIG
from oops.transform import Transform

class Frame(object):
    """A Frame is an abstract class that returns a Transform (rotation matrix
    and spin vector) given a Scalar time.

    A Transform converts from a reference coordinate frame to a target
    coordinate frame, either of which could be non-inertial. All coordinate
    frames are ultimately referenced to J2000.
    """

    # To avoid circular imports; filled in by oops/__init__.py
    EVENT_CLASS = None
    PATH_CLASS = None

    J2000 = None
    WAYFRAME_REGISTRY = {}
    FRAME_CACHE = {}
    TEMPORARY_FRAME_ID = 10000

    STANDARD_FRAMES = set()     # Frames that always have the same definition

    ############################################################################
    # Each subclass must override...
    ############################################################################

    def __init__(self):
        """Constructor for a Frame object.

        Every frame must have these attributes:

            frame_id    the string ID of this frame.
            wayframe    the wayframe that uniquely identifies this frame. For
                        registered frames, this is the Wayframe with the same
                        ID, as it appears in the WAYFRAME_REGISTRY dictionary.
                        If a frame is not registered, then its wayframe
                        attribute should point to itself.
            reference   the reference frame as a registered wayframe. A frame
                        rotates coordinates from the reference frame into the
                        given frame.
            origin      the origin path as a waypoint. It identifies the path
                        relative to which this frame is defined; None if the
                        frame is inertial.
            shape       the shape of the Frame object, as a tuple of dimensions.
                        This is the shape of the Transform object returned by
                        frame.transform_at_time() for a single value of time.
            keys        a set of keys by which this frame is cached.

        The primary definition of a frame will be assigned this attribute by the
        registry:

            ancestry    a list of Frame objects beginning with this frame and
                        ending with with J2000, where each Frame in sequence is
                        the reference of the previous frame:
                            self.ancestry[0] = self.
                            self.ancestry[1] = reference frame of self.
                            ...
                            self.ancestry[-1] = J2000.
                        Note that an immediate ancestor must either be an
                        inertial frame, or else use the same origin.

            wrt_j2000   a definition of the same frame relative to the J2000
                        coordinate frame.
        """

        pass

    #===========================================================================
    def transform_at_time(self, time, quick={}):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates must be given relative to
        the center of rotation.

        Input:
            time        a Scalar time.
            quick       an optional dictionary of parameter values to use as
                        overrides to the configured default QuickPath and
                        QuickFrame parameters; use False to disable the use of
                        QuickPaths and QuickFrames.

        Return:         the corresponding Tranform applicable at the specified
                        time(s). The transform rotates vectors from the
                        reference frame to this frame.

        Note that the time and the Frame object are not required to have the
        same shape; standard rules of broadcasting apply.
        """

        raise NotImplementedError(type(self).__name__ + '.transform_at_time ' +
                                  'is not implemented')

    #===========================================================================
    def transform_at_time_if_possible(self, time, quick={}):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates must be given relative to
        the center of rotation.

        Unlike method transform_at_time(), this variant tolerates times that
        raise cspyce errors. It returns a new time Scalar along with the new
        Transform, where both objects skip over the times at which the transform
        could not be evaluated.

        Default behavior is to assume that all times are valid. As a result,
        this function calls transform_at_time, but also returns the given time
        Scalar. This behavior is overridden by SpiceFrame, where occasional
        short gaps in a C-kernel can be tolerated as long as a QuickFrame
        interpolates across them.

        Input:
            time            a Scalar time, which must be 0-D or 1-D.

        Return:             (newtimes, transform)
            newtimes        a Scalar time, possibly containing a subset of the
                            times given.
            transform       the corresponding Tranform applicable at the new
                            time(s).
        """

        time = Scalar.as_scalar(time)
        return (time, self.transform_at_time(time, quick=quick))

    #===========================================================================
    def node_at_time(self, time, quick={}):
        """Angle from the frame's X-axis to the X-Y plane's ascending node on
        the J2000 equator.
        """

        frame = self.wrt(Frame.J2000)
        xform = frame.transform_at_time(time, quick=quick)
        z_axis_wrt_j2000 = xform.unrotate(Vector3.ZAXIS)
        (x,y,_) = z_axis_wrt_j2000.to_scalars()

        if (x,y) == (0.,0.):
            return Scalar(0.)

        return (y.arctan2(x) + Scalar.HALFPI) % Scalar.TWOPI

    ############################################################################
    # String operations
    ############################################################################

    @property
    def reference_id(self):
        return self.reference.frame_id

    @property
    def origin_id(self):
        if self.origin is None:
            return None

        return self.origin.path_id

    def __str__(self):
        return (type(self).__name__ + '(' + self.frame_id + '/' +
                                            self.reference_id + ')')

    def __repr__(self):
        return self.__str__()

    ############################################################################
    # Registry Management
    ############################################################################

    # A frame can be registered by an ID string. Any frame so registered can be
    # retrieved afterward from the registry using the string. However, it is not
    # necessary to register a frame.
    #
    # When an ID is registered for the first time, a Wayframe is constructed and
    # added to the WAYFRAME_REGISTRY, which is a dictionary that returns the
    # Wayframe associated with any ID.
    #
    # Wayframe is a Frame subclass that contains no information except the ID.
    # You can use this subclass anywhere a Frame is required. This makes it
    # possible to identify a frame without indicating how it is to be
    # calculated. That information is in the registry and will be used to
    # determine the method of calculation when the time comes.
    #
    # Normally, a frame is defined by name only once. It can be overridden if
    # necessary, but note that this might alter the behavior of any other object
    # that refers to this frame by name. Objects that link directly to a Frame
    # object, rather than referring to it by name or Wayframe, will still refer
    # to the original object, but the old frame will no longer be accessible
    # from the registry.
    #
    # The FRAME_CACHE contains every calculated version of a Frame object. This
    # saves us the effort of re-connecting a frame (wayframe, reference, origin)
    # each time is is needed. The FRAME_CACHE is keyed as follows:
    #      wayframe
    #      (wayframe, reference_wayframe)
    #
    # If the key is not a tuple, then this constitutes the primary definition of
    # the frame. The reference wayframe must already be in the cache, and the
    # new wayframe cannot be in the cache.
    #
    # The frame registry can also contain "shortcuts". By default, a frame
    # calculation might require a sequence of internal calculations to get from
    # a given origin to a given target frame. For example, the frame of
    # of Saturn's rings relative to HST could involve internal calculations of
    # Saturn's rings relative to Saturn, Saturn relative to J2000, and HST
    # relative to J2000. A shortcut is a way to define a more direct
    # calculation.

    @staticmethod
    def initialize_registry():
        """Initialize the frame registry.

        It is not generally necessary to call this function directly.
        """

        # After first call, return
        if Frame.WAYFRAME_REGISTRY:
            return

        # Initialize the WAYFRAME_REGISTRY
        Frame.WAYFRAME_REGISTRY[None] = Frame.J2000
        Frame.WAYFRAME_REGISTRY['J2000'] = Frame.J2000

        # Initialize the FRAME_CACHE
        Frame.J2000.keys = {Frame.J2000, (Frame.J2000, Frame.J2000)}
        for key in Frame.J2000.keys:
            Frame.FRAME_CACHE[key] = Frame.J2000

    #===========================================================================
    @staticmethod
    def reset_registry():
        """Reset the registry to its initial state. Mainly useful for debugging.
        """

        Frame.WAYFRAME_REGISTRY.clear()
        Frame.FRAME_CACHE.clear()
        Frame.initialize_registry()

    #===========================================================================
    def register(self, shortcut=None, override=False, unpickled=False):
        """Register a Frame's definition.

        A shortcut makes it possible to calculate one SPICE frame relative to
        another without calculating all the intermediate frames. If a shortcut
        name is given, then this frame is treated as a shortcut definition. The
        frame is cached under the shortcut name and also under the tuple
        (wayframe, reference_wayframe).

        If override is True, then this frame will override the current primary
        definition of any previous frame with the same name. The old frame might
        still exist, but it will not be available from the registry.

        If unpickled is True and a frame with the same ID is already in the
        registry, then this frame is not registered. Instead, its wayframe will
        be defined by the frame with the same name that is already registered.

        If the frame ID is None, blank, or begins with '.', it is treated as a
        temporary path and is not registered.
        """

        # Make sure the registry is initialized
        if Frame.J2000 is None:
            Frame.initialize_registry()

        frame_id = self.frame_id

        # Handle a shortcut
        if shortcut is not None:
            if shortcut in Frame.FRAME_CACHE:
                Frame.FRAME_CACHE[shortcut].keys -= {shortcut}
            Frame.FRAME_CACHE[shortcut] = self
            self.keys |= {shortcut}

            key = (Frame.FRAME_CACHE[frame_id], self.reference)
            if key in Frame.FRAME_CACHE:
                Frame.FRAME_CACHE[key].keys -= {key}
            Frame.FRAME_CACHE[key] = self
            self.keys |= {key}

            return

        # Fill in a temporary name if needed; don't register
        if self.frame_id in (None, '', '.'):
            self.frame_id = Frame.temporary_frame_id()
            self.wayframe = self
            return

        # Don't register a name beginning with dot
        if self.frame_id.startswith('.'):
            self.wayframe = self
            return

        # Raise ValueError if the reference frame is unregistered
        if self.reference.frame_id not in Frame.WAYFRAME_REGISTRY:
            raise ValueError('frame ' + self.frame_id +
                             ' cannot be registered because it connects to ' +
                             'unregistered frame ' + self.reference.frame_id)

        # Raise KeyError if the reference is not a WayFrame
        _ = Frame.FRAME_CACHE[self.reference]

        # If the ID is not registered, insert this as the primary definition
        if (frame_id not in Frame.WAYFRAME_REGISTRY) or override:

            # Fill in the ancestry
            reference = Frame.as_primary_frame(self.reference)
            self.ancestry = [reference] + reference.ancestry

            # Register the Wayframe
            wayframe = Wayframe(frame_id, self.origin, self.shape)
            self.wayframe = wayframe
            Frame.WAYFRAME_REGISTRY[frame_id] = wayframe

            # Cache the frame under two keys
            self.keys = {wayframe, (wayframe, self.reference)}
            for key in self.keys:
                Frame.FRAME_CACHE[key] = self

            # Cache the wayframe
            wayframe.keys = {(wayframe, wayframe)}
            for key in wayframe.keys:
                Frame.FRAME_CACHE[key] = wayframe

            # Also define the frame with respect to J2000
            if self.reference == Frame.J2000:
                self.wrt_j2000 = self
            else:
                self.wrt_j2000 = self.wrt(Frame.J2000)

                key = (wayframe, Frame.J2000)
                self.wrt_j2000.keys = {key}
                Frame.FRAME_CACHE[key] = self.wrt_j2000

        # Otherwise, just insert a secondary definition
        else:
            if not hasattr(self, 'wayframe') or self.wayframe is None:
                self.wayframe = Frame.WAYFRAME_REGISTRY[frame_id]

            # If this is not an unpickled frame, make it the frame returned by
            # any of the standard keys.
            if not unpickled:
                # Cache (self.wayframe, self.reference); overwrite if necessary
                key = (self.wayframe, self.reference)
                if key in Frame.FRAME_CACHE:        # remove an old version
                    Frame.FRAME_CACHE[key].keys -= {key}

                Frame.FRAME_CACHE[key] = self
                self.keys |= {key}

    #===========================================================================
    @staticmethod
    def as_frame(frame):
        """The Frame object given the registered name or the object itself."""

        if frame is None:
            return None
        if isinstance(frame, Frame):
            return frame

        return Frame.WAYFRAME_REGISTRY[frame]

    #===========================================================================
    @staticmethod
    def as_primary_frame(frame):
        """The primary definition of a Frame object."""

        if frame is None:
            return None

        if not isinstance(frame, Frame):
            frame = Frame.WAYFRAME_REGISTRY[frame]

        return Frame.FRAME_CACHE[frame.wayframe]

    #===========================================================================
    @staticmethod
    def as_wayframe(frame):
        """The wayframe given a Frame or ID."""

        if frame is None:
            return None
        if isinstance(frame, Frame):
            return frame.wayframe

        return Frame.WAYFRAME_REGISTRY[frame]

    #===========================================================================
    @staticmethod
    def as_frame_id(frame):
        """The frame ID given the object or a registered ID."""

        if frame is None:
            return None

        if isinstance(frame, Frame):
            return frame.frame_id

        return frame

    #===========================================================================
    @staticmethod
    def temporary_frame_id():
        """A temporary frame ID. This is assigned once and never re-used.
        """

        while True:
            Frame.TEMPORARY_FRAME_ID += 1
            frame_id = 'TEMPORARY_' + str(Frame.TEMPORARY_FRAME_ID)

            if frame_id not in Frame.WAYFRAME_REGISTRY:
                return frame_id

    #===========================================================================
    def is_registered(self):
        """True if this frame is registered."""

        return (self.frame_id in Frame.WAYFRAME_REGISTRY)

    ############################################################################
    # Frame Generators
    ############################################################################

    # Can be overridden by some classes such as SpiceFrame, where it is easier
    # to make connections between frames.
    def wrt(self, reference):
        """Construct a Frame that transforms from a reference frame to this one.

        Input:
            reference   a reference Frame object or its wayframe or ID.
        """

        # Convert a Wayframe to its registered version
        # Sorry for the ugly use of 'self'!
        if isinstance(self, Wayframe):
            self = Frame.WAYFRAME_REGISTRY[self.frame_id]

        # Convert the reference to a frame
        reference = Frame.as_frame(reference)

        # Deal with an unregistered frame
        if not self.is_registered():
            if self.reference == reference.wayframe:
                return self

            if reference.reference == self.wayframe:
                return ReversedFrame(reference)

            return LinkedFrame(self, self.reference.wrt(reference))

        # Use the cache if possible
        key = (self.wayframe, reference.wayframe)
        if key in Frame.FRAME_CACHE:
            return Frame.FRAME_CACHE[key]

        # Look up the primary target definition
        try:
            target = Frame.FRAME_CACHE[self.wayframe]
        except KeyError:
            # On failure, link from the reference frame
            return LinkedFrame(self, self.reference.wrt(reference))

        # Look up the primary reference definition
        try:
            reference = Frame.FRAME_CACHE[reference.wayframe]
        except KeyError:
            # On failure, link through the reference's reference
            return RelativeFrame(target.wrt(reference.reference),
                                 reference)

        # If the target is an ancestor of the reference, reverse the direction
        # and try again
        if target in reference.ancestry:
            newframe = reference.wrt(target)
            return ReversedFrame(newframe)

        # Otherwise, search from the parent frame and then link
        newframe = target.ancestry[0].wrt(reference)
        return LinkedFrame(target, newframe)

    #===========================================================================
    def quick_frame(self, time, quick={}):
        """A QuickFrame that approximates this frame within given time limits.

        A QuickFrame operates by sampling the given frame and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed
        up performance when the same frame must be evaluated repeatedly many
        times, e.g., for every pixel of an image.

        Input:
            time        a Scalar defining the set of times at which the frame is
                        to be evaluated.
            quick       if False, no QuickPath is created and self is returned;
                        if a dictionary, then the values provided override the
                        values in the default dictionary QUICK.dictionary, and
                        the merged dictionary is used.
        """

        OVERHEAD = 500      # Assume it takes the equivalent time of this many
                            # evaluations just to set up the QuickFrame.
        SPEEDUP = 5.        # Assume that evaluations are this much faster once
                            # the QuickFrame is set up.
        SAVINGS = 0.2       # Require at least a 20% savings in evaluation time.

        # Make sure a QuickFrame has been requested
        if not isinstance(quick, dict):
            return self

        # These subclasses do not require QuickFrames
        if type(self) in (QuickFrame, Wayframe, AliasFrame):
            return self

        # Obtain the local QuickFrame dictionary
        quickdict = QUICK.dictionary
        if len(quick) > 0:
            quickdict = quickdict.copy()
            quickdict.update(quick)

        if not quickdict['use_quickframes']:
            return self

        # Determine the time interval
        if type(time) in (list,tuple):
            (tmin, tmax, count) = time
        else:
            time = Scalar.as_scalar(time)
            tmin = time.min()
            tmax = time.max()
            count = np.size(time.values)

        if tmin == Scalar.MASKED:
            return self

        if isinstance(tmin, Scalar):
            tmin = tmin.values
            tmax = tmax.values

        # If QuickFrames already exists...
        if not hasattr(self, 'quickframes'):
            self.quickframes = []

        # If the whole time range is already covered, just return this one
        for quickframe in self.quickframes:
            if tmin >= quickframe.t0 and tmax <= quickframe.t1:

                if LOGGING.quickframe_creation:
                    LOGGING.diagnostic('Re-using QuickFrame: ' + str(self),
                                       '(%.3f, %.3f)' % (tmin, tmax))

                return quickframe

        # See if the overhead would make any work justified
        if count < OVERHEAD:
            return self

        # Get dictionary parameters
        extension = quickdict['frame_time_extension']
        dt = quickdict['frame_time_step']
        extras = quickdict['frame_extra_steps']

        # Extend the time domain
        tmin -= extension
        tmax += extension

        # See if any QuickFrame can be efficiently extended
        for quickframe in self.quickframes:

            # If there's no overlap, skip it
            if (quickframe.t0 > tmax + dt) or (quickframe.t1 < tmin - dt):
                continue

            # Otherwise, check the effort involved
            duration = (max(tmax, quickframe.t1) - min(tmin, quickframe.t0))
            steps = int(duration // dt) - quickframe.steps

            effort_extending_quickframe = OVERHEAD + steps + count/SPEEDUP
            if count >= effort_extending_quickframe:
                if LOGGING.quickframe_creation:
                    LOGGING.diagnostic('Extending QuickFrame: ' + str(self),
                                       '(%.3f, %.3f)' % (tmin, tmax))

                quickframe.extend((tmin,tmax))
                return quickframe

        # Evaluate the effort using a QuickFrame compared to the effort without
        steps = int((tmax - tmin) // dt) + 2*extras
        effort_using_quickframe = OVERHEAD + steps + count/SPEEDUP
        if count < (1. + SAVINGS) * effort_using_quickframe:
            return self

        if LOGGING.quickframe_creation:
            LOGGING.diagnostic('New QuickFrame: ' + str(self),
                               '(%.3f, %.3f)' % (tmin, tmax))

        result = QuickFrame(self, (tmin, tmax), quickdict)

        if len(self.quickframes) > quickdict['quickframe_cache']:
            self.quickframes = [result] + self.quickframes[:-1]
        else:
            self.quickframes = [result] + self.quickframes

        return result

################################################################################
# Required Subclasses
################################################################################

class Wayframe(Frame):
    """Wayframe is lightweight Frame subclass used to identify a registered
    Frame by its ID.

    A Wayframe does not provide information about how it is to be calculated. It
    has the property that self.origin == self, so it always evaluates to a null
    transform.

    A Wayframe cannot be registered.
    """

    def __init__(self, frame_id, origin=None, shape=()):
        """Constructor for a Wayframe.

        Input:
            frame_id    the frame ID to use for both the target and reference.
            origin      the waypoint of this frame's origin.
            shape       shape of the path.
        """

        # Required attributes
        self.wayframe  = self
        self.frame_id  = frame_id
        self.reference = self
        self.origin    = origin
        self.shape     = shape
        self.keys      = set()

    def __getstate__(self):
        # A frame might not get assigned the same ID on the next run of OOPS, so
        # saving the name alone is not meaningful. Instead, we save the current
        # primary definition, which has the same frame_id, origin, and shape.
        return (Frame.as_primary_frame(self),)

    def __setstate__(self, state):
        (primary_frame,) = state

        # As a side-effect, we have just un-pickled the primary frame that this
        # waypoint previously represented.
        self.__init__(primary_frame.frame_id, primary_frame.origin,
                      primary_frame.shape)

    def transform_at_time(self, time, quick={}):
        return Transform(Matrix3.IDENTITY, Vector3.ZERO, self, self,
                                                         self.origin)

    def register(self):
        return      # does nothing

    def __str__(self):
        return 'Wayframe(' + self.frame_id + ')'

################################################################################

class AliasFrame(Frame):
    """An AliasFrame takes on the properties of the frame it is given.

    Used to create a quick, temporary frame that transforms from an arbitrary
    frame to this one. An AliasFrame cannot be registered.
    """

    def __init__(self, frame):

        self.alias = Frame.as_frame(frame)

        # Required attributes
        self.wayframe  = self.alias.wayframe
        self.frame_id  = self.alias.frame_id
        self.reference = self.alias.reference
        self.origin    = self.alias.origin
        self.shape     = ()
        self.keys      = set()

    def __getstate__(self):
        return (self.alias,)

    def __setstate__(self, state):
        self.__init__(*state)

    def register(self):
        raise TypeError('an AliasFrame cannot be registered')

    def transform_at_time(self, time, quick={}):
        return self.alias.transform_at_time(time, quick=quick)

    def transform_at_time_if_possible(self, time, quick={}):
        return self.alias.transform_at_time_if_possible(time, quick=quick)

################################################################################

class LinkedFrame(Frame):
    """A LinkedFrame applies one frame's transform to another.

    The new frame describes coordinates in one frame relative to the reference
    of the second frame.
    """

    def __init__(self, frame, parent):
        """Constructor for a LinkedFrame.

        Input:
            frame       a frame, which must be define relative to the given
                        parent.
            parent      a frame to which the above will be linked.
        """

        self.frame  = Frame.as_frame(frame)
        self.parent = Frame.as_frame(parent)

        if self.frame.reference != self.parent.wayframe:
            raise ValueError('LinkedFrame mismatch: %s, %s'
                             % (self.frame.reference, self.parent.wayframe))

        # Required attributes
        self.wayframe  = self.frame.wayframe
        self.frame_id  = self.wayframe.frame_id
        self.reference = self.parent.reference
        self.shape     = Qube.broadcasted_shape(self.frame.shape,
                                                self.parent.shape)
        self.keys      = set()

        if self.frame.origin is None:
            self.origin = self.parent.origin
        elif self.parent.origin is None:
            self.origin = self.frame.origin
        else:
            self.origin = self.parent.origin
            # assert self.frame.origin == self.parent.origin

        if frame.is_registered() and parent.is_registered():
            self.register()     # save for later use

    def __getstate__(self):
        return (self.frame, self.parent)

    def __setstate__(self, state):
        self.__init__(*state)

    def transform_at_time(self, time, quick={}):

        parent = self.parent.transform_at_time(time, quick=quick)
        xform = self.frame.transform_at_time(time, quick=quick)

        return xform.rotate_transform(parent)

    def transform_at_time_if_possible(self, time, quick={}):

        (time1, parent) = self.parent.transform_at_time_if_possible(time)
        (time2, xform) = self.frame.transform_at_time_if_possible(time1)

        if time1.shape != time2.shape:
            parent = self.parent.transform_at_time(time2)

        return (time2, xform.rotate_transform(parent))

################################################################################

class RelativeFrame(Frame):
    """A Frame that returns the one frame times the inverse of another. The
    two frames must have a common reference. The combined frame converts
    coordinates from the second frame to the first.
    """

    def __init__(self, frame1, frame2):

        self.frame1 = Frame.as_frame(frame1)
        self.frame2 = Frame.as_frame(frame2)

        if self.frame1.reference != self.frame2.reference:
            raise ValueError('RelativeFrame mismatch: %s, %s'
                             % (self.frame1.reference, self.frame2.reference))

        # Required attributes
        self.wayframe  = self.frame1.wayframe
        self.frame_id  = self.wayframe.frame_id
        self.reference = self.frame2.wayframe
        self.shape     = Qube.broadcasted_shape(self.frame1, self.frame2)
        self.keys      = set()

        # Identify the origin; confirm compatibility
        self.origin = frame1.origin
        if self.origin is None:
            self.origin = frame2.origin
        elif frame2.origin is not None:
            self.origin = frame2.origin
            # assert frame2.origin == self.origin

        if frame1.is_registered() and frame2.is_registered():
            self.register()     # save for later use

    def __getstate__(self):
        return (self.frame1, self.frame2)

    def __setstate__(self, state):
        self.__init__(*state)

    def transform_at_time(self, time, quick={}):

        xform1 = self.frame1.transform_at_time(time)
        xform2 = self.frame2.transform_at_time(time)

        return xform1.rotate_transform(xform2.invert())

    def transform_at_time_if_possible(self, time, quick=None):

        (time1, xform1) = self.frame1.transform_at_time_if_possible(time)
        (time2, xform2) = self.frame2.transform_at_time_if_possible(time1)

        if time1.shape != time2.shape:
            xform1 = self.frame1.transform_at_time(time2)

        return (time2, xform1.rotate_transform(xform2.invert()))

################################################################################

class ReversedFrame(Frame):
    """A Frame that generates the inverse Transform of a given Frame."""

    def __init__(self, frame):

        self.oldframe = Frame.as_frame(frame)

        # Required attributes
        self.wayframe  = frame.reference
        self.frame_id  = frame.reference.frame_id
        self.reference = frame.wayframe
        self.origin    = frame.origin
        self.shape     = frame.shape
        self.keys      = set()

        if frame.is_registered() and frame.reference.is_registered():
            self.register()         # save for later use

    def __getstate__(self):
        return (self.oldframe)

    def __setstate__(self, state):
        self.__init__(*state)

    def transform_at_time(self, time, quick={}):
        return self.oldframe.transform_at_time(time).invert()

    def transform_at_time_if_possible(self, time, quick={}):
        (time, xform) = self.oldframe.transform_at_time_if_possible(time)
        return (time, xform.invert())

################################################################################

class QuickFrame(Frame):
    """QuickFrame is a Frame subclass that returns Transform objects based on
    interpolation of another Frame within a specified time window.
    """

    def __init__(self, frame, interval, quickdict):
        """Constructor for a QuickFrame.

        Input:
            frame           the Frame object that this Frame will emulate.
            interval        a tuple containing the start and stop times to use,
                            in TDB seconds.
            quickdict       a dictionary containing all the QuickFrame
                            parameters.
        """

        if frame.shape != ():
            raise ValueError('shape of QuickFrame must be ()')

        self.slowframe = frame

        self.wayframe  = frame.wayframe
        self.frame_id  = frame.frame_id
        self.reference = frame.reference
        self.origin    = frame.origin
        self.shape     = ()
        self.keys      = set()

        self.t0 = interval[0]
        self.t1 = interval[1]
        self.dt = quickdict['frame_time_step']

        self.extras = quickdict['frame_extra_steps']
        self.times = np.arange(self.t0 - self.extras * self.dt,
                               self.t1 + self.extras * self.dt + self.dt,
                               self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]
        self.steps = int((self.t1 - self.t0) / self.dt + 0.5)

        (new_times, self.transforms) = \
                    self.slowframe.transform_at_time_if_possible(self.times)
        self.times = new_times.values

        self.quickdict = quickdict
        self.omega_numerical = quickdict['quickframe_numerical_omega']
        self.omega_zero = quickdict['ignore_quickframe_omega']
        self.omega_fixed = (self.transforms.omega.shape == ())

        self._spline_setup()

        # Test the precision
        self.precision_self_check = quickdict['frame_self_check']
        if self.precision_self_check is not None:
            t = self.times[:-1] + self.dt/2.        # Halfway points

            true_transform = self.slowframe.transform_at_time(t)
            (matrix, omega) = self._interpolate_matrix_omega(t)

            dmatrix = (Matrix(true_transform.matrix) - Matrix(matrix)).rms()

            domega = (true_transform.omega - omega).rms()
            if true_transform.omega.rms() != 0.:
                domega /= true_transform.omega.rms()

            error = max(np.max(dmatrix.vals), np.max(domega.vals))
            if error > self.precision_self_check:
                raise ValueError('precision tolerance not achieved: ' +
                                  str(error) + ' > ' +
                                  str(self.precision_self_check))

    def __getstate__(self):
        if PICKLE_CONFIG.quickframe_details:
            return self.__dict__
        else:
            interval = (self.t0, self.t1)
            quickdict = {
                'frame_time_step'           : self.dt,
                'frame_extra_steps'         : self.extras,
                'quickframe_numerical_omega': self.omega_numerical,
                'ignore_quickframe_omega'   : self.omega_zero,
                'frame_self_check'          : self.precision_self_check,
            }
            return (self.slowpath, interval, quickdict)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self.__init__(*state)
        else:
            self.__dict__ = state

    ####################################

    def transform_at_time(self, time, quick=False):
        (matrix, omega) = self._interpolate_matrix_omega(time)
        return Transform(matrix, omega, self, self.reference, self.origin)

    def transform_at_time_if_possible(self, time, quick=False):
        xform = self.transform_at_time(time, quick=False)
        return (time, xform)

    def register(self):
        raise TypeError('a QuickFrame cannot be registered')

    ############################################################################
    # This new version uses quaternions.
    ############################################################################

    def _spline_setup(self):

        KIND = 3

        # Create splines for all four components of the quaternion
        quaternions = Quaternion.as_quaternion(self.transforms.matrix)
        self.quat_splines = np.empty((4,), dtype='object')
        for i in range(4):
          self.quat_splines[i] = InterpolatedUnivariateSpline(self.times,
                                    quaternions.vals[...,i],
                                    k=KIND)

        # Don't interpolate omega if frame is inertial
        if self.omega_zero or self.omega_fixed:
          self.omega_splines = None
          self.qdot_splines = None

        # Create derivative splines if omega solution is numerical
        elif self.omega_numerical:
          self.omega_splines = None
          self.qdot_splines = np.empty((4,), dtype='object')
          for i in range(4):
            self.qdot_splines[i] = self.quat_splines[i].derivative(1)

        # Otherwise, create splines for the vector components of omega
        else:
          self.omega_splines = np.empty((3,), dtype='object')
          self.qdot_splines = None
          for i in range(3):
            self.omega_splines[i] = InterpolatedUnivariateSpline(self.times,
                                        self.transforms.omega.vals[...,i],
                                        k=KIND)

    #===========================================================================
    def _interpolate_matrix_omega(self, time, collapse_threshold=None):

        if collapse_threshold is None:
            collapse_threshold = self.quickdict[
                                    'quickframe_linear_interpolation_threshold']

        # time can only be a 1-D array in the splines
        time = Scalar.as_scalar(time)
        tflat = time.flatten()
        if np.size(tflat.vals) == 0:
            identity = np.zeros(time.shape + (3,3))
            identity[...,0,0] = 1.
            identity[...,1,1] = 1.
            identity[...,2,2] = 1.
            matrix = Matrix3(identity, True)
            omega = Vector3(np.ones(time.shape + (3,)), True)
            return (matrix, omega)

        tflat_max = np.max(tflat.vals)
        tflat_min = np.min(tflat.vals)
        time_diff = tflat_max - tflat_min

        # Case 1: A single time
        if time_diff == 0.:
            quat = np.empty((4,))
            quat[0] = self.quat_splines[0](tflat_max)
            quat[1] = self.quat_splines[1](tflat_max)
            quat[2] = self.quat_splines[2](tflat_max)
            quat[3] = self.quat_splines[3](tflat_max)

            quat = Quaternion(quat)

            matrix_vals = np.empty(tflat.shape + (3,3))
            matrix_vals[...,:,:] = Matrix3.as_matrix3(quat).vals
            matrix = Matrix3(matrix_vals)

            if self.omega_splines is not None:
                om = np.empty((3,))
                om[0] = self.omega_splines[0](tflat_max)
                om[1] = self.omega_splines[1](tflat_max)
                om[2] = self.omega_splines[2](tflat_max)

                omega_vals = np.empty(tflat.shape + (3,))
                omega_vals[...,:] = om[:]
                omega = Vector3(omega_vals)

            elif self.qdot_splines is not None:
                qd = np.empty((4,))
                qd[0] = self.qdot_splines[0](tflat_max)
                qd[1] = self.qdot_splines[1](tflat_max)
                qd[2] = self.qdot_splines[2](tflat_max)
                qd[3] = self.qdot_splines[3](tflat_max)
                qdot = Quaternion(qd)

                omega_vals = np.empty(tflat.shape + (3,))
                omega_vals[...,:] = 2. * (qdot / quat).values[1:4]
                omega = Vector3(omega_vals)

            elif self.omega_zero:
                omega = Vector3.ZERO

            else:
                omega = self.transforms.omega

        # Case 2: Use linear interpolation for a brief enough time span
        elif time_diff < collapse_threshold:
            frac = (tflat.vals - tflat_min) / time_diff

            # Create a time scalar just containing the end points
            tflat2 = Scalar([tflat_min, tflat_max])

            quat = np.empty((2,4))
            quat[:,0] = self.quat_splines[0](tflat2.vals)
            quat[:,1] = self.quat_splines[1](tflat2.vals)
            quat[:,2] = self.quat_splines[2](tflat2.vals)
            quat[:,3] = self.quat_splines[3](tflat2.vals)

            quat = Quaternion(quat[0] + (quat[1] - quat[0])
                                        * frac[...,np.newaxis])
            matrix = Matrix3.as_matrix3(quat)

            if self.omega_splines is not None:
                om = np.empty((2,3))
                om[:,0] = self.omega_splines[0](tflat2.vals)
                om[:,1] = self.omega_splines[1](tflat2.vals)
                om[:,2] = self.omega_splines[2](tflat2.vals)

                omega = Vector3(om[0] + frac[...,np.newaxis] * (om[1] - om[0]))

            elif self.qdot_splines is not None:
                qd = np.empty((2,4))
                qd[:,0] = self.qdot_splines[0](tflat2.vals)
                qd[:,1] = self.qdot_splines[1](tflat2.vals)
                qd[:,2] = self.qdot_splines[2](tflat2.vals)
                qd[:,3] = self.qdot_splines[3](tflat2.vals)
                qd_x2 = 2. * qd

                qdot_x2 = Quaternion(qd_x2[0] + frac[...,np.newaxis]
                                                * (qd_x2[1] - qd_x2[0]))

                omega = (qdot_x2 / quat).to_parts()[1]

            elif self.omega_zero:
                omega = Vector3.ZERO

            else:
                omega = self.transforms.omega

        # Case 3: Use spline evaluation
        else:
            quat = np.empty(tflat.shape + (4,))
            quat[...,0] = self.quat_splines[0](tflat.vals)
            quat[...,1] = self.quat_splines[1](tflat.vals)
            quat[...,2] = self.quat_splines[2](tflat.vals)
            quat[...,3] = self.quat_splines[3](tflat.vals)

            quat = Quaternion(quat)
            matrix = Matrix3.as_matrix3(quat)

            if self.omega_splines is not None:
                om = np.empty(tflat.shape + (3,))
                om[...,0] = self.omega_splines[0](tflat.vals)
                om[...,1] = self.omega_splines[1](tflat.vals)
                om[...,2] = self.omega_splines[2](tflat.vals)

                omega = Vector3(om)

            elif self.qdot_splines is not None:
                qd = np.empty(tflat.shape + (4,))
                qd[...,0] = self.qdot_splines[0](tflat.vals)
                qd[...,1] = self.qdot_splines[1](tflat.vals)
                qd[...,2] = self.qdot_splines[2](tflat.vals)
                qd[...,3] = self.qdot_splines[3](tflat.vals)

                qdot = Quaternion(qd)
                omega = 2. * (qdot / quat).to_parts()[1]

            elif self.omega_zero:
                omega = Vector3.ZERO

            else:
                omega = self.transforms.omega

        # Return the matrices and rotation vectors
        matrix = matrix.reshape(time.shape)

        if omega.shape != ():
            omega = omega.reshape(time.shape)

        return (matrix, omega)

    #===========================================================================
    def extend(self, interval):
        """Modify the given QuickFrame to accommodate the given time interval.
        """

        # If the interval fits inside already, we're done
        if interval[0] >= self.t0 and interval[1] <= self.t1:
            return

        # Extend the interval
        if interval[0] < self.t0:
            count0 = int((self.t0 - interval[0]) // self.dt) + 1 + self.extras
            times0 = np.arange(count0) * self.dt + self.t0 - count0 * self.dt
            (times0,
             transform0) = self.slowframe.transform_at_time_if_possible(times0,
                                                                    quick=False)
            count0 = len(times0)
        else:
            count0 = 0

        if interval[1] > self.t1:
            count1 = int((interval[1] - self.t1) // self.dt) + 1 + self.extras
            times1 = np.arange(count1) * self.dt + self.t1 + self.dt
            (times1,
             transform1) = self.slowframe.transform_at_time_if_possible(times1,
                                                                    quick=False)
            count1 = len(times1)
        else:
            count1 = 0

        if count0 + count1 == 0:
            return

        # Allocate the new arrays
        old_size = self.times.size
        new_size = old_size + count0 + count1

        times = np.empty(new_size)
        matrix_vals = np.empty((new_size,3,3))

        if self.omega_fixed:
            omega_vals = self.transforms.omega.vals
        else:
            omega_vals = np.empty((new_size,3))

        # Copy the new arrays
        if count0 > 0:
            times[0:count0] = times0.vals
            matrix_vals[0:count0,:,:] = transform0.matrix.vals
            if not self.omega_fixed:
                omega_vals[0:count0,:] = transform0.omega.vals

        times[count0:count0+old_size] = self.times
        matrix_vals[count0:count0+old_size,:,:] = self.transforms.matrix.vals
        if not self.omega_fixed:
            omega_vals[count0:count0+old_size,:] = self.transforms.omega.vals

        if count1 > 0:
            times[count0+old_size:] = times1.vals
            matrix_vals[count0+old_size:,:,:] = transform1.matrix.vals
            if not self.omega_fixed:
                omega_vals[count0+old_size:,:] = transform1.omega.vals

        # Generate the new transforms
        self.times = times
        self.t0 = self.times[0]
        self.t1 = self.times[-1]
        self.steps = int((self.t1 - self.t0) / self.dt + 0.5)

        new_transforms = Transform(Matrix3(matrix_vals),
                                   Vector3(omega_vals),
                                   self.transforms.frame,
                                   self.transforms.reference)
        self.transforms = new_transforms

        # Update the splines
        self._spline_setup()

################################################################################
# Initialization at load time...
################################################################################

# Initialize Frame.J2000
Frame.J2000 = Wayframe('J2000')
Frame.J2000.ancestry = []
Frame.J2000.wrt_j2000 = Frame.J2000

# Initialize the registry
Frame.initialize_registry()
Frame.STANDARD_FRAMES.add(Frame.J2000)

################################################################################
