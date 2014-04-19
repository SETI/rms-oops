################################################################################
# oops/frame_/frame.py: Abstract class Frame and its required subclasses
################################################################################

import numpy as np
import scipy.interpolate as interp
from polymath import *

from oops.config     import QUICK, LOGGING
from oops.transform  import Transform
import oops.utils    as utils

class Frame(object):
    """A Frame is an abstract class that returns a Transform (rotation matrix
    and spin vector) given a Scalar time. A Transform converts from a reference
    coordinate frame to a target coordinate frame, either of which could be
    non-inertial. All coordinate frame are ultimately referenced to J2000."""

    J2000 = None
    WAYFRAME_REGISTRY = {}
    FRAME_CACHE = {}
    TEMPORARY_FRAME_ID = 10000

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

        pass

    @property
    def reference_id(self): return self.reference.frame_id

    @property
    def origin_id(self): return self.origin.path_id

    # string operations
    def __str__(self):
        return (type(self).__name__ + '(' + self.frame_id + '/' +
                                            self.reference_id + ')')

    def __repr__(self): return self.__str__()

    ############################################################################
    # Registry Management
    ############################################################################

    # A frame can be registered by an ID string. Any frame so registered can be
    # retrieved afterward from the registry using the string. However, it is not
    # necessary to register a frame; just set the attribute 'wayframe' to self
    # instead.
    #
    # When an ID is registered for the first time, a Wayframe is constructed and
    # added to the WAYFRAME_REGISTRY, which is a dictionary that returns the
    # Wayframe associated with any ID.
    #
    # If a frame is redefined, a new Wayframe is created and replaces the old
    # one in the registry. This enables us to identify objects that are no
    # longer using an up-to-date version of a frame.
    #
    # The FRAME_CACHE contains every calculated version of a Frame object. This
    # saves us the effort of re-connecting a frame (wayframe, reference, origin)
    # each time is is needed. The FRAME_CACHE is keyed as follows:
    #       wayframe
    #       (wayframe, reference_wayframe)
    #
    # If the key is not a tuple, then this constitutes the primary definition of
    # the frame. The reference wayframe must already be in the cache, and the
    # new wayframe cannot be in the cache.

    @staticmethod
    def initialize_registry():
        """Initialize the frame registry.

        It is not generally necessary to call this function directly."""

        # After first call, return
        if Frame.WAYFRAME_REGISTRY: return

        # Initialize the WAYFRAME_REGISTRY
        Frame.WAYFRAME_REGISTRY[None] = Frame.J2000
        Frame.WAYFRAME_REGISTRY['J2000'] = Frame.J2000

        # Initialize the FRAME_CACHE
        Frame.J2000.keys = {Frame.J2000, (Frame.J2000, Frame.J2000)}
        for key in Frame.J2000.keys:
            Frame.FRAME_CACHE[key] = Frame.J2000

    @staticmethod
    def reset_registry():
        """Reset the registry to its initial state. Mainly useful for debugging.
        """

        Frame.WAYFRAME_REGISTRY.clear()
        Frame.FRAME_CACHE.clear()
        Frame.initialize_registry()

    def register(self, shortcut=None):
        """Register a Frame's definition.

        A shortcut makes it possible to calculate one SPICE frame relative to
        another without calculating all the intermediate frames. If a shortcut
        name is given, then this frame is treated as a shortcut definition. The
        frame is cached under the shortcut name and also under the tuple
        (wayframe, reference_wayframe).

        If the frame ID is None, blank, or begins with '.', it is treated as a
        temporary path and is not registered.
        """

        # Make sure the registry is initialized
        if Frame.J2000 is None: Frame.initialize_registry()

        WAYFRAME_REG = Frame.WAYFRAME_REGISTRY
        FRAME_CACHE = Frame.FRAME_CACHE

        id = self.frame_id

        # Handle a shortcut
        if shortcut is not None:
            if shortcut in FRAME_CACHE: FRAME_CACHE[shortcut].keys -= {shortcut}
            FRAME_CACHE[shortcut] = self
            self.keys |= {shortcut}

            key = (FRAME_CACHE[id], self.reference)
            if key in FRAME_CACHE: FRAME_CACHE[key].keys -= {key}
            FRAME_CACHE[key] = self
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

        # Make sure the reference frame is registered; otherwise raise KeyError
        try:
            test = WAYFRAME_REG[self.reference.frame_id]
        except KeyError:
            raise ValueError('frame ' + self.frame_id +
                             ' cannot be registered because it connects to ' +
                             'unregistered frame ' + self.reference.frame_id)

        test = FRAME_CACHE[self.reference]

        # If the ID is not registered, insert this as the primary definition
        if id not in WAYFRAME_REG:

            # Fill in the ancestry
            reference = Frame.as_primary_frame(self.reference)
            self.ancestry = [reference] + reference.ancestry

            # Register the Wayframe
            wayframe = Wayframe(id, self.origin, self.shape)
            self.wayframe = wayframe
            WAYFRAME_REG[id] = wayframe

            # Cache the frame under two keys
            self.keys = {wayframe, (wayframe, self.reference)}
            for key in self.keys:
                FRAME_CACHE[key] = self

            # Cache the wayframe
            wayframe.keys = {(wayframe, wayframe)}
            for key in wayframe.keys:
                FRAME_CACHE[key] = wayframe

            # Also define the frame with respect to J2000
            if self.reference == Frame.J2000:
                self.wrt_j2000 = self
            else:
                self.wrt_j2000 = self.wrt(Frame.J2000)

                key = (wayframe, Frame.J2000)
                self.wrt_j2000.keys = {key}
                FRAME_CACHE[key] = self.wrt_j2000

        # Otherwise, just insert a secondary definition
        else:
            if not hasattr(self, 'wayframe') or self.wayframe is None:
                self.wayframe = WAYFRAME_REG[id]

            # Cache (self.wayframe, self.reference); overwrite if necessary
            key = (self.wayframe, self.reference)
            if key in FRAME_CACHE:          # remove an old version
                FRAME_CACHE[key].keys -= {key}

            FRAME_CACHE[key] = self
            self.keys |= {key}

    @staticmethod
    def as_frame(frame):
        """Return a Frame object given the registered name or the object itself.
        """

        if frame is None: return None

        if isinstance(frame, Frame): return frame

        return Frame.WAYFRAME_REGISTRY[frame]

    @staticmethod
    def as_primary_frame(frame):
        """Return the primary definition of a Frame object."""

        if frame is None: return None

        if not isinstance(frame, Frame):
            frame = Frame.WAYFRAME_REGISTRY[frame]

        return Frame.FRAME_CACHE[frame.wayframe]

    @staticmethod
    def as_wayframe(frame):
        """Return the wayframe given a Frame or ID."""

        if frame is None: return None

        if isinstance(frame, Frame):
            return frame.wayframe

        return Frame.WAYFRAME_REGISTRY[frame]

    @staticmethod
    def as_frame_id(path):
        """Return a frame ID given the object or a registered ID."""

        if frame is None: return None

        if isinstance(frame, Frame):
            return frame.frame_id

        return frame

    @staticmethod
    def temporary_frame_id():
        """Return a temporary frame ID. This is assigned once and never re-used.
        """

        while True:
            Frame.TEMPORARY_FRAME_ID += 1
            frame_id = "TEMPORARY_" + str(Frame.TEMPORARY_FRAME_ID)

            if frame_id not in Frame.WAYFRAME_REGISTRY:
                return frame_id

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

    ########################################

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
        if type(quick) != dict: return self

        # These subclasses do not require QuickFrames
        if type(self) in (QuickFrame, Wayframe, AliasFrame): return self

        # Obtain the local QuickFrame dictionary
        quickdict = QUICK.dictionary
        if len(quick) > 0:
            quickdict = quickdict.copy()
            quickdict.update(quick)

        if not quickdict['use_quickframes']: return slowpath

        # Determine the time interval
        if type(time) in (list,tuple):
            (tmin, tmax, count) = time
        else:
            time = Scalar.as_scalar(time)
            tmin = time.min()
            tmax = time.max()
            count = np.size(time.values)

        if tmin == Scalar.MASKED: return self

        # If QuickFrames already exists...
        if not hasattr(self, 'quickframes'):
            self.quickframes = []

        # If the whole time range is already covered, just return this one
        for quickframe in self.quickframes:
            if tmin >= quickframe.t0 and tmax <= quickframe.t1:

                if LOGGING.quickframe_creation:
                    print LOGGING.prefix, "Re-using QuickFrame: " + str(self),
                    print '(%.3f, %.3f)' % (tmin, tmax)

                return quickframe

        # See if the overhead would make any work justified
        if count < OVERHEAD: return self

        # Get dictionary parameters
        extension = quickdict['frame_time_extension']
        dt = quickdict['frame_time_step']
        extras = quickdict['frame_extra_steps']

        # Extend the time domain
        tmin -= extension
        tmax += extension

        # See if a QuickFrame can be efficiently extended
        for quickframe in self.quickframes:
            duration = (max(tmax, quickframe.t1) - min(tmin, quickframe.t0))
            steps = int(duration//dt) - quickframe.times.size

            # Compare the effort involved in extending to the effort without
            effort_extending_quickframe = OVERHEAD + steps + count/SPEEDUP
            if count >= effort_extending_quickframe: 

                if LOGGING.quickframe_creation:
                    print LOGGING.prefix, "Extending QuickFrame: " + str(self),
                    print '(%.3f, %.3f)' % (tmin, tmax)

                quickframe.extend((tmin,tmax))
                return quickframe

        # Evaluate the effort using a QuickFrame compared to the effort without
        steps = int((tmax - tmin)//dt) + 2*extras
        effort_using_quickframe = OVERHEAD + steps + count/SPEEDUP
        if count < (1. + SAVINGS) * effort_using_quickframe:
            return self

        if LOGGING.quickframe_creation:
            print LOGGING.prefix, "New QuickFrame: " + str(self),
            print '(%.3f, %.3f)' % (tmin, tmax)

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
    """Wayframe is used to identify a Frame ID in the registry.

    When evaluated, it always returns a null transform. A Wayframe cannot be
    registered by the user."""

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

    def transform_at_time(self, time, quick={}):
        return Transform(Matrix3.IDENTITY, Vector3.ZERO, self, self,
                                                         self.origin)

    def register(self): return      # does nothing

    def __str__(self): return "Wayframe(" + self.frame_id + ")"

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

    def register(self):
        raise TypeError('an AliasFrame cannot be registered')

    def transform_at_time(self, time, quick={}):
        return self.alias.transform_at_time(time, quick=quick)

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

        assert self.frame.reference == self.parent.wayframe

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

    def transform_at_time(self, time, quick={}):

        parent = self.parent.transform_at_time(time, quick=quick)
        xform = self.frame.transform_at_time(time, quick=quick)
        return xform.rotate_transform(parent)

        return transform

################################################################################

class RelativeFrame(Frame):
    """A Frame that returns the one frame times the inverse of another. The
    two frames must have a common reference. The combined frame converts
    coordinates from the second frame to the first."""

    def __init__(self, frame1, frame2):

        self.frame1 = frame1
        self.frame2 = frame2

        assert self.frame1.reference == self.frame2.reference

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

    def transform_at_time(self, time, quick={}):

        xform1 = self.frame1.transform_at_time(time)
        xform2 = self.frame2.transform_at_time(time)

        return xform1.rotate_transform(xform2.invert())

################################################################################

class ReversedFrame(Frame):
    """A Frame that generates the inverse Transform of a given Frame.
    """

    def __init__(self, frame):

        self.oldframe = frame

        # Required attributes
        self.wayframe  = frame.reference
        self.frame_id  = frame.reference.frame_id
        self.reference = frame.wayframe
        self.origin    = frame.origin
        self.shape     = frame.shape
        self.keys      = set()

        if frame.is_registered() and frame.reference.is_registered():
            self.register()         # save for later use

    def transform_at_time(self, time, quick={}):

        return self.oldframe.transform_at_time(time).invert()

################################################################################

class QuickFrame(Frame):
    """QuickFrame is a Frame subclass that returns Transform objects based on
    interpolation of another Frame within a specified time window."""

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
        self.dt = quickdict["frame_time_step"]

        self.extras = quickdict["frame_extra_steps"]
        self.times = np.arange(self.t0 - self.extras * self.dt,
                               self.t1 + self.extras * self.dt + self.dt,
                               self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        self.transforms = self.slowframe.transform_at_time(self.times)
        self._spline_setup()

        # Test the precision
        precision_self_check = quickdict["frame_self_check"]
        if precision_self_check is not None:
            t = self.times[:-1] + self.dt/2.        # Halfway points

            true_transform = self.slowframe.transform_at_time(t)
            (matrix, omega) = self._interpolate_matrix_omega(t)

            dmatrix = (true_transform.matrix - matrix).rms()

            domega = (true_transform.omega - omega).rms()
            if true_transform.omega.rms() != 0.:
                domega /= true_transform.omega.rms()

            error = max(np.max(dmatrix.vals), np.max(domega.vals))
            if error > precision_self_check:
                raise ValueError("precision tolerance not achieved: " +
                                  str(error) + " > " +
                                  str(precision_self_check))

    ####################################

    def transform_at_time(self, time, quick=False):
        (matrix, omega) = self._interpolate_matrix_omega(time)
        return Transform(matrix, omega, self, self.reference, self.origin)

    def register(self):
        raise TypeError('a QuickFrame cannot be registered')

    ####################################

    def _spline_setup(self):

        # This would be faster with quaternions but I'm lazy
        KIND = 3
        self.matrix = np.empty((3,3), dtype="object")
        # for i in range(3):
        for i in range(2):
          for j in range(3):
            self.matrix[i,j] = interp.UnivariateSpline(self.times,
                                    self.transforms.matrix.vals[...,i,j],
                                    k=KIND)

        # Don't interpolate omega if frame is inertial
        if self.transforms.omega == Vector3.ZERO:
            self.omega = None
        else:
            self.omega = np.empty((3,), dtype="object")
            for i in range(3):
                self.omega[i] = interp.UnivariateSpline(self.times,
                                        self.transforms.omega.vals[...,i],
                                        k=KIND)

    def _interpolate_matrix_omega(self, time):

        # time can only be a 1-D array in the splines
        tflat = Scalar.as_scalar(time).flatten()
        matrix = np.empty(list(tflat.shape) + [3,3])
        omega  = np.zeros(list(tflat.shape) + [3])

        # Evaluate the matrix and rotation vector
        matrix[...,0,0] = self.matrix[0,0](tflat.vals)
        matrix[...,0,1] = self.matrix[0,1](tflat.vals)
        matrix[...,0,2] = self.matrix[0,2](tflat.vals)
        matrix[...,1,0] = self.matrix[1,0](tflat.vals)
        matrix[...,1,1] = self.matrix[1,1](tflat.vals)
        matrix[...,1,2] = self.matrix[1,2](tflat.vals)
        # matrix[...,2,0] = self.matrix[2,0](tflat.vals)
        # matrix[...,2,1] = self.matrix[2,1](tflat.vals)
        # matrix[...,2,2] = self.matrix[2,2](tflat.vals)

        if self.omega is not None:
            omega[...,0] = self.omega[0](tflat.vals)
            omega[...,1] = self.omega[1](tflat.vals)
            omega[...,2] = self.omega[2](tflat.vals)

        # Normalize the matrix
        matrix[...,2,:] = utils.ucross3d(matrix[...,0,:], matrix[...,1,:])
        matrix[...,0,:] = utils.ucross3d(matrix[...,1,:], matrix[...,2,:])
        matrix[...,1,:] = utils.unit(matrix[...,1,:])

        # Return the positions and velocities
        return (Matrix3(matrix).reshape(time.shape),
                Vector3(omega).reshape(time.shape))

    ####################################

    def extend(self, interval):
        """Modify the given QuickFrame to accommodate the given time interval.
        """

        # If the interval fits inside already, we're done
        if interval[0] >= self.t0 and interval[1] <= self.t1: return

        # Extend the interval
        if interval[0] < self.t0:
            count0 = int((self.t0 - interval[0]) // self.dt) + 1 + self.extras
            new_t0 = self.t0 - count0 * self.dt
            times  = np.arange(count0) * self.dt + new_t0
            transform0 = self.slowframe.transform_at_time(times, quick=False)
        else:
            count0 = 0
            new_t0 = self.t0

        if interval[1] > self.t1:
            count1 = int((interval[1] - self.t1) // self.dt) + 1 + self.extras
            new_t1 = self.t1 + count1 * self.dt
            times  = np.arange(count1) * self.dt + self.t1 + self.dt
            transform1 = self.slowframe.transform_at_time(times, quick=False)
        else:
            count1 = 0
            new_t1 = self.t1

        # Allocate the new arrays
        old_size = self.times.size
        new_size = old_size + count0 + count1
        matrix_vals = np.empty((new_size,3,3))
        omega_vals = np.empty((new_size,3))

        # Copy the new arrays
        if count0 > 0:
            matrix_vals[0:count0,:,:] = transform0.matrix.vals
            omega_vals[0:count0,:] = transform0.omega.vals

        matrix_vals[count0:count0+old_size,:,:] = self.transforms.matrix.vals
        omega_vals[count0:count0+old_size,:] = self.transforms.omega.vals

        if count1 > 0:
            matrix_vals[count0+old_size:,:,:] = transform1.matrix.vals
            omega_vals[count0+old_size:,:] = transform1.omega.vals

        # Generate the new transforms
        self.times = np.arange(new_t0, new_t1 + self.dt/2., self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

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

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Frame(unittest.TestCase):

    def runTest(self):

        # Re-import here to so modules all come from the oops tree
        from oops.frame_.frame import Frame, QuickFrame, ReversedFrame

        # More imports are here to avoid conflicts
        import os
        import cspice
        from oops.frame_.rotation import Rotation
        from oops.frame_.spiceframe import SpiceFrame
        from oops.path_.spicepath import SpicePath
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/naif0009.tls"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/pck00010.tpc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

        Frame.reset_registry()

        # QuickFrame tests

        ignore = SpicePath("EARTH", "SSB")
        ignore = SpicePath("MOON", "SSB")
        ignore = SpiceFrame("IAU_EARTH", "J2000")
        moon  = SpiceFrame("IAU_MOON", "IAU_EARTH")
        quick = QuickFrame(moon, (-5.,5.),
                        dict(QUICK.dictionary, **{"frame_self_check":3.e-14}))

        # Perfect precision is impossible
        try:
            quick = QuickFrame(moon, (-5.,5.),
                        dict(QUICK.dictionary, **{"frame_self_check":0.}))
            self.assertTrue(False, "No ValueError raised for PRECISION = 0.")
        except ValueError: pass

        # Timing tests...
        test = np.zeros(200000)
        # ignore = moon.transform_at_time(test, quick=False)  # takes about 10 sec
        ignore = quick.transform_at_time(test) # takes way less than 1 sec

        Frame.reset_registry()

        ################################
        # Test unregistered frames
        ################################

        j2000 = Frame.as_wayframe('J2000')
        rot_180 = Rotation(np.pi, 2, j2000)
        self.assertTrue(rot_180.frame_id.startswith('TEMPORARY'))

        xform = rot_180.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], -1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], -1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        rot_neg60 = Rotation(-np.pi/3, 2, rot_180)
        self.assertTrue(rot_neg60.frame_id.startswith('TEMPORARY'))

        c60 = 0.5
        s60 = np.sqrt(0.75)

        xform = rot_neg60.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0],  c60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], -s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1],  c60, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        rot_neg120 = Rotation(-np.pi/1.5, 2, rot_neg60)
        self.assertTrue(rot_neg120.frame_id.startswith('TEMPORARY'))

        xform = rot_neg120.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], -c60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], -s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], -c60, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        # Attempt to register a frame defined relative to an unregistered frame
        self.assertRaises(ValueError, Rotation, -np.pi, 2, rot_neg60, 'NEG180')

        # Link unregistered frame to registered frame
        identity = rot_neg120.wrt('J2000')

        xform = identity.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], 1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], 1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        # Link registered frame to unregistered frame
        identity = Frame.J2000.wrt(rot_neg120)

        xform = identity.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], 1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], 1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        # Link unregistered frame to registered frame
        identity = rot_neg120.wrt(rot_180)

        xform = identity.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], -1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], -1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
