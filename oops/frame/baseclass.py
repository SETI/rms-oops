################################################################################
# oops/frame/baseclass.py: Abstract class Frame and its required subclasses
#
# 2/13/12 Modified (MRS) - implemented and tested QuickFrame.
################################################################################

import numpy as np
import scipy.interpolate as interp

import oops.frame.registry as registry
from oops.xarray.all import *
from oops.transform import Transform

class Frame(object):
    """A Frame is an abstract class that returns a Transform (rotation matrix
    and spin vector) given a Scalar time. A Transform converts from a reference
    coordinate frame to a target coordinate frame, either of which could be
    non-inertial. All coordinate frame are ultimately referenced to J2000.

    Every frame must have these attributes:
        frame_id        the ID of this frame, either a string or an integer.
        reference_id    the ID of the reference frame, either a string or an
                        integer. A frame rotates coordinates from the reference
                        frame into the given frame.
        origin_id       the ID of the origin relative to which the frame is
                        defined; None if it is inertial.
        shape           the shape of the Frame object, as a list of dimensions.
                        This is the shape of the Transform object returned by
                        frame.transform_at_time() for a single value of time.

    The primary definition of a frame will be assigned this attribute by the
    registry:

    ancestry            a list of Frame objects beginning with this frame and
                        ending with with J2000, where each Frame in sequence is
                        the reference of the previous frame:
                            self.ancestry[0] = self.
                            self.ancestry[1] = reference frame of self.
                            ...
                            self.ancestry[-1] = J2000.
    """

    # Change to False for better accuracy but slower performance, possibly
    # MUCH slower performance
    USE_QUICKFRAMES = True

########################################
# Each subclass must override...
########################################

    def __init__(self):
        """Constructor for a Frame object."""

        pass

    def transform_at_time(self, time):
        """Returns a Transform object that rotates coordinates from the
        reference frame into this frame as a function of time.

        Input:
            time    a Scalar time.

        Return:     the corresponding Tranform applicable at the specified
                    time(s). The transform rotates vectors from the reference
                    frame to this frame.

        Note that the time and the Frame object are not required to have the
        same shape; standard rules of broadcasting apply.
        """

        pass

################################################################################
# Registry Management
################################################################################

    # The global registry is keyed in two ways. If the key is a string, then
    # this constitutes the primary definition of the frame. The reference_id
    # of the frame must already be in the registry, and the new frame's ID
    # cannot be in the registry.

    # We also create secondary definitions of a frame, where it is defined
    # relative to a different reference frame. These are entered into the
    # registry keyed by a tuple: (frame_id, reference_id). This saves the
    # effort of re-creating a frame used repeatedly.

    @staticmethod
    def initialize_registry():
        """Initializes the registry. It is not generally necessary to call this
        function, but it can be used to reset the registry for purposes of
        debugging."""

        if registry.J2000 is None:
            registry.J2000 = Null("J2000")
            registry.J2000.ancestry = [registry.J2000]

        registry.REGISTRY = {"J2000": registry.J2000,
                             ("J2000","J2000"): registry.J2000}

    def register(self, shortcut=None):
        """Registers a Frame definition. If the frame's ID is new, it is assumed
        to be the primary definition and is keyed by the ID alone. However, a
        primary definition must use a reference ID that is already registered.

        Otherwise or in addition, a secondary key is added to the registry if it
        is not already present:
            (frame_id, reference_id)
        This key also points to the same Frame object.

        If a shortcut name is given, then self is treated as a shortcut
        definition. The frame is registered under the shortcut name and also
        under the tuple (frame_id, reference_id), but other registered
        definitions of the path are not modified.
        """

        # Make sure the reference frame is registered; raise KeyError on failure
        reference = registry.REGISTRY[self.reference_id]

        # Handle a shortcut
        if shortcut is not None:
            registry.REGISTRY[shortcut] = self
            registry.REGISTRY[(self.frame_id, self.reference_id)] = self
            return

        # If the ID is unregistered, insert this as a primary definition
        try:
            test = registry.REGISTRY[self.frame_id]
        except KeyError:
            registry.REGISTRY[self.frame_id] = self

            # Fill in the ancestry too
            self.ancestry = [self] + reference.ancestry

            # Also save the Null Frame
            registry.REGISTRY[(self.frame_id,
                               self.frame_id)] = Null(self.frame_id)

            # Also define the path with respect to J2000 if possible
            if self.reference_id != "J2000":
              try:
                wrt_j2000 = self.connect_to("J2000")
                registry.REGISTRY[(wrt_j2000.frame_id, "J2000")] = wrt_j2000
              except: pass

        # Insert or replace the secondary definition for
        # (self.frame_id, self.reference_id) as a secondary definition
        registry.REGISTRY[(self.frame_id, self.reference_id)] = self

    def unregister(self):
        """Removes this frame from the registry."""

        # Import only occurs when needed
        import oops.path.registry as path_registry

        # Note that we only delete the primary entry and any frame in which this
        # is one of the end points. If the frame is used as an intermediate step
        # between other frames, it will cease to be visible in the dictionary
        # but frames that use it will continue to function unchanged. However,
        # it is safest to remove all usage of a frame at the time it is
        # unregistered.

        frame_id = self.frame_id
        for key in registry.REGISTRY.keys():
            if frame_id == key: del registry.REGISTRY[key]

            if type(key) == type(()):
                if frame_id == key[0]:
                    del registry.REGISTRY[key]
                elif frame_id == key[1]: 
                    del registry.REGISTRY[key]

        path_registry.unregister_frame(frame_id)

    def reregister(self):
        """Adds this frame to the registry, replacing any definition of the same
        name."""

        self.unregister()
        self.register()

################################################################################
# Event operations
################################################################################

# These must be defined in Event.py and not here, because that would create a
# circular dependency in the order that modules are loaded.

    def rotate_event(self, event):
        """Returns the same event at the same origin, but rotated forward into a
        new coordinate frame.

        Input:
            event       Event object to rotate.
        """

        assert self.reference_id == event.frame_id
        assert self.origin_id == event.origin_id

        return self.transform_at_time(event.time).rotate_event(event)

    def unrotate_event(self, event):
        """Returns the same event at the same origin, but unrotated backward
        into the reference coordinate frame.

        Input:
            event       Event object to un-rotate.
        """

        assert self.frame_id == event.frame_id
        assert self.origin_id == event.origin_id

        return self.transform_at_time(event.time).unrotate_event(event)

################################################################################
# Frame Generator
################################################################################

    @staticmethod
    def connect(target, reference):
        """Returns a Frame object that transforms from the given reference Frame
        to the given target Frame.

        Input:
            target      a Frame object or the registered ID of the destination
                        frame.
            reference   a Frame object or the registered ID of the starting frame.

        The shape of the connected frame will will be the result of broadcasting
        the shapes of the target and reference.
        """

        # Convert to IDs
        target_id    = registry.as_id(target)
        reference_id = registry.as_id(reference)

        # If the path already exists, just return it
        try:
            return registry.REGISTRY[(target_id, reference_id)]
        except KeyError: pass

        # Otherwise, construct it from the common ancestor...

        target_frame = registry.REGISTRY[target_id]
        return target_frame.connect_to(reference_id)

    # Can be overridden by some classes such as SpiceFrame, where it is easier
    # to make connections.
    def connect_to(self, reference):
        """Returns a Frame object that transforms from an arbitrary reference
        frame to this frame.

        Input:
            reference       a reference Frame object or its registered name.
        """

        # Find the fundamental frame definitions
        target    = registry.as_primary(self)
        reference = registry.as_primary(reference)

        # Check for compatibility of rotating frames. The origins must match.
        if target.origin_id is not None and reference.origin_id is not None:
            assert target.origin_id == reference.origin_id

        # Find the common ancestry
        (target_ancestry,
         reference_ancestry) = Frame.common_ancestry(target, reference)
        # print Frame.str_ancestry((target_ancestry, origin_ancestry))

        # We can ignore the final (matching) entry in each list
        target_ancestry = target_ancestry[:-1]
        reference_ancestry = reference_ancestry[:-1]

        # Look up or construct the target's frame from the common ancestor
        if target_ancestry == []:
            target_frame = None
        elif len(target_ancestry) == 1:
            target_frame = target
        else:
            try:
                target_frame = registry.lookup((target.frame_id,
                                              target_ancestry[-1].reference_id))
            except KeyError:
                target_frame = Linked(target_ancestry)
                target_frame.register()

        # Look up or construct the common ancestor's frame from the reference
        if reference_ancestry == []:
            reference_frame = None
        elif len(reference_ancestry) == 1:
            reference_frame = reference
        else:
            try:
                reference_frame = registry.lookup(
                                        (reference.frame_id,
                                         reference_ancestry[-1].reference_id))
            except KeyError:
                reference_frame = Linked(reference_ancestry)
                reference_frame.register()

        # Return the relative frame
        if reference_frame is None:
            if target_frame is None:
                result = Null(target.frame_id)
            else:
                result = target_frame
        else:
            if target_frame is None:
                result = Inverse(reference_frame)
            else:
                result = Relative(target_frame, reference_frame)

        result.register()
        return result

    @staticmethod
    def common_ancestry(frame1, frame2):
        """Returns a pair of ancestry lists for the two given frames, where both
        lists end at Frames with the same name."""

        # Identify the first common ancestor of both frames
        for i in range(len(frame1.ancestry)):
            id1 = frame1.ancestry[i].frame_id

            for j in range(len(frame2.ancestry)):
                id2 = frame2.ancestry[j].frame_id

                if id1 == id2:
                    return (frame1.ancestry[:i+1], frame2.ancestry[:j+1])

        return (frame1.ancestry, frame2.ancestry)       # should never happen

    @staticmethod
    def str_ancestry(tuple):        # For debugging
        list = ["(["]

        for item in tuple:
            for frame in item:
                list += [frame.frame_id,"\", \""]

            list.pop()
            list += ["\"], [\""]

        list.pop()
        list += ["\"])"]

        return "".join(list)

########################################
# Arithmetic operators
########################################

    # unary "~" operator
    def __inverse__(self):
        return frame.Inverse(self)

    # binary "*" operator
    def __mul__(self, arg):
        return frame.Product(self, arg)

    # binary "/" operator
    def __div__(self, arg):
        return frame.Relative(self, arg)

    # string operations
    def __str__(self):
        return "Frame(" + self.frame_id + "/" + self.reference_id + ")"

########################################

    def quick_frame(self, epoch, quick_info=None):
        """Returns a new QuickFrame object that provides accurate approximations
        to the transform returned by this frame. It can speed up performance
        substantially when the same frame must be evaluated repeatedly but
        within a very narrow range of times.

        Input:
            epoch           the mid-time in TDB of the interpolation.
            quick_info      a dictionary containing the additional optional
                            parameters:
              ["WINDOW"]    the full duration of the window within which path
                            information should be available, in seconds. Default
                            is 20.
              ["SAMPLING"]  the sampling interval of the spline, in seconds.
                            Default is 0.01.
              ["PRECISION"] the fractional precision required of the returned
                            interpolated values; this is tested when the
                            QuickFrame is created, and a ValueError is raised
                            if the requirement is not met. Default is None,
                            in which case the precision is not tested.
        """

        if quick_info is None: return self
        if not Frame.USE_QUICKFRAMES: return self

        return QuickFrame(self, epoch, quick_info)

################################################################################
# Required Subclasses
################################################################################

class Inverse(Frame):
    """A Frame that generates the inverse Transform of a given Frame.
    """

    def __init__(self, frame):

        self.oldframe = registry.as_frame(frame)

        # Required fields
        self.frame_id     = self.oldframe.reference_id
        self.reference_id = self.oldframe.frame_id
        self.origin_id    = self.oldframe.origin_id

        self.shape = self.oldframe.shape

    def transform_at_time(self, time):

        return self.oldframe.transform_at_time(time).invert()

################################################################################

class Linked(Frame):
    """A Frame that links together a list of Frame objects, so that the
    Transform converts from the reference frame of the last entry to the
    coordinate frame of the first entry. The reference_id of each list entry
    must be the frame_id of the entry that follows.
    """

    def __init__(self, frames):
        """Constructor for a LinkedFrame.

        Input:
            frames      a list of connected frames. The reference_id of each
                        frame must be the frame_id of the one that follows.
        """

        self.frames = frames

        # Required fields
        self.frame_id     = frames[0].frame_id
        self.reference_id = frames[-1].reference_id

        # Identify the origin; confirm compatibility
        self.origin_id = None
        for frame in frames:
            if frame.origin_id is not None:
                if self.origin_id is None:
                    self.origin_id = frame.origin_id
                else:
                    assert self.origin_id == frame.origin_id

        self.shape = Array.broadcast_shape(tuple(self.frames))

    def transform_at_time(self, time):

        transform = self.frames[-1].transform_at_time(time)
        for frame in self.frames[-2::-1]:
           transform = frame.transform_at_time(time).rotate_transform(transform)

        return transform

################################################################################

class Null(Frame):
    """A Frame that always returns a null transform."""

    def __init__(self, frame_id, origin_id=None):

        # Required fields
        self.frame_id     = frame_id
        self.reference_id = frame_id
        self.origin_id    = None

        self.shape = []

    def transform_at_time(self, time):

        return Transform.null_transform(self.frame_id)

    def __str__(self): return "Frame(" + self.frame_id + ")"

################################################################################

class Relative(Frame):
    """A Frame that returns the one frame times the inverse of another. The
    two frames must have a common reference. The combined frame converts
    coordinates from the second frame to the first."""

    def __init__(self, frame1, frame2):

        self.frame1 = frame1
        self.frame2 = frame2

        assert self.frame1.reference_id == self.frame2.reference_id

        # Required fields
        self.frame_id     = self.frame1.frame_id
        self.reference_id = self.frame2.frame_id

        # Identify the origin; confirm compatibility
        self.origin_id = frame1.origin_id
        if self.origin_id is None:
            self.origin_id = frame2.origin_id
        elif frame2.origin_id is not None:
            assert frame2.origin_id == self.origin_id

        self.shape = Array.broadcast_shape((self.frame1, self.frame2))

    def transform_at_time(self, time):

        return self.frame1.transform_at_time(time).rotate_transform(
               self.frame2.transform_at_time(time).invert())

################################################################################

class QuickFrame(Frame):
    """QuickFrame is a Frame subclass that returns Transform objects based on
    interpolation of another Frame within a specified time window."""

    def __init__(self, frame, epoch, quick_info={}):
        """Constructor for a QuickFrame.

        Input:
            frame           the Frame object that this Frame will emulate.
            epoch           the mid-time in TDB of the interpolation.
            quick_info      a dictionary containing the additional optional
                            parameters:
              ["WINDOW"]    the full duration of the window within which path
                            information should be available, in seconds. Default
                            is 20.
              ["SAMPLING"]  the sampling interval of the spline, in seconds.
                            Default is 0.01.
              ["PRECISION"] the fractional precision required of the returned
                            interpolated values; this is tested when the
                            QuickFrame is created, and a ValueError is raised
                            if the requirement is not met. Default is None,
                            in which case the precision is not tested.
        """

        self.frame = frame
        self.frame_id     = frame.frame_id
        self.reference_id = frame.reference_id
        self.origin_id    = frame.origin_id

        assert frame.shape == []
        self.shape = []

        if quick_info is None: quick_info == {}
        self.info = quick_info

        self.epoch = epoch
        self.window = 20.
        self.dt = 0.01
        self.precision = None

        try:
            self.window = self.info["WINDOW"]
        except KeyError: pass

        try:
            self.dt = self.info["SAMPLING"]
        except KeyError: pass

        try:
            self.precision = self.info["PRECISION"]
        except KeyError: pass

        self.t0 = self.epoch - self.window/2.
        self.t1 = self.epoch + self.window/2.

        EXPAND = 4
        self.times = np.arange(self.t0 - EXPAND * self.dt,
                               self.t1 + EXPAND * self.dt + self.dt, self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        self.transforms = self.frame.transform_at_time(self.times)

        # This would be faster with quaternions but I'm lazy
        KIND = 3
        self.matrix = np.empty((3,3), dtype="object")
        for i in range(3):
          for j in range(3):
            self.matrix[i,j] = interp.UnivariateSpline(self.times,
                                    self.transforms.matrix.vals[...,i,j],
                                    k=KIND)

        self.omega = np.empty((3,), dtype="object")
        for i in range(3):
            self.omega[i] = interp.UnivariateSpline(self.times,
                                    self.transforms.omega.vals[...,i],
                                    k=KIND)

        # Test the precision
        if self.precision is not None:
            t = self.times[:-1] + self.dt/2.        # Halfway points

            true_transform = self.frame.transform_at_time(t)

            matrix = np.empty(t.shape + (3,3))
            omega  = np.empty(t.shape + (3,))

            # Evaluate the matrix and rotation vector
            matrix[...,0,0] = self.matrix[0,0](t)
            matrix[...,0,1] = self.matrix[0,1](t)
            matrix[...,0,2] = self.matrix[0,2](t)
            matrix[...,1,0] = self.matrix[1,0](t)
            matrix[...,1,1] = self.matrix[1,1](t)
            matrix[...,1,2] = self.matrix[1,2](t)
            matrix[...,2,0] = self.matrix[2,0](t)
            matrix[...,2,1] = self.matrix[2,1](t)
            matrix[...,2,2] = self.matrix[2,2](t)

            omega[...,0] = self.omega[0](t)
            omega[...,1] = self.omega[1](t)
            omega[...,2] = self.omega[2](t)

            # Normalize the matrix
            matrix[...,0,:] = utils.ucross3d(matrix[...,1,:], matrix[...,2,:])
            matrix[...,1,:] = utils.ucross3d(matrix[...,2,:], matrix[...,0,:])
            matrix[...,2,:] = utils.unit(matrix[...,2,:])

            dmatrix = abs(true_transform.matrix - matrix)

            domega = abs(true_transform.omega - omega)
            if abs(true_transform.omega) != 0.:
                domega /= abs(true_transform.omega)

            error = max(np.max(dmatrix.vals), np.max(domega.vals))
            if error > self.precision:
                raise ValueError("precision tolerance not achieved: " +
                                  str(error) + " > " + str(self.precision))

    def transform_at_time(self, time):

        # time can only be a 1-D array
        time   = Scalar.as_scalar(time)
        matrix = np.empty((time.vals.size, 3, 3))
        omega  = np.empty((time.vals.size, 3))

        # Evaluate the matrix and rotation vector
        t = time.vals.ravel()

        matrix[...,0,0] = self.matrix[0,0](t)
        matrix[...,0,1] = self.matrix[0,1](t)
        matrix[...,0,2] = self.matrix[0,2](t)
        matrix[...,1,0] = self.matrix[1,0](t)
        matrix[...,1,1] = self.matrix[1,1](t)
        matrix[...,1,2] = self.matrix[1,2](t)
        matrix[...,2,0] = self.matrix[2,0](t)
        matrix[...,2,1] = self.matrix[2,1](t)
        matrix[...,2,2] = self.matrix[2,2](t)

        omega[...,0] = self.omega[0](t)
        omega[...,1] = self.omega[1](t)
        omega[...,2] = self.omega[2](t)

        # Normalize the matrix
        matrix[...,0,:] = utils.ucross3d(matrix[...,1,:], matrix[...,2,:])
        matrix[...,1,:] = utils.ucross3d(matrix[...,2,:], matrix[...,0,:])
        matrix[...,2,:] = utils.unit(matrix[...,2,:])

        # Return the Event
        return Transform(matrix.reshape(time.vals.shape + (3,)),
                         omega.reshape(time.vals.shape + (3,)),
                         self.frame_id, self.reference_id, self.origin_id)

    def __str__(self):
        return "QuickFrame(" + self.frame_id + "/" + self.reference_id + ")"

################################################################################
# Initialize the registry
################################################################################

registry.FRAME_CLASS = Frame
Frame.initialize_registry()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Frame(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        from oops.frame.spiceframe import SpiceFrame
        from oops.path.spicepath   import SpicePath
        import oops.path.registry  as path_registry

        registry.initialize_registry()
        path_registry.initialize_registry()

        # QuickFrame tests
        ignore = SpicePath("EARTH", "SSB")
        ignore = SpicePath("MOON", "SSB")
        ignore = SpiceFrame("IAU_EARTH", "J2000")
        moon  = SpiceFrame("IAU_MOON", "IAU_EARTH")
        quick = QuickFrame(moon, 0., {"PRECISION": 3.e-14}) # confirms precision

        # Perfect precision is impossible
        try:
            quick = QuickFrame(moon, 0., {"PRECISION": 0.})
            self.assertTrue(False, "No ValueError raised for PRECISION = 0.")
        except ValueError: pass

        # Timing tests...
        # test = np.zeros(1000000)
        # ignore = moon.transform_at_time(test)  # takes about 50 sec
        # ignore = quick.transform_at_time(test) # takes way less than 1 sec

        registry.initialize_registry()
        path_registry.initialize_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
