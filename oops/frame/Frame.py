import numpy as np
import unittest

import oops

################################################################################
# Frame
################################################################################

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

    OOPS_CLASS = "Frame"

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

########################################
# Override for enhanced performance
########################################

    def quick_frame(self, epoch, span, precision=None, check=True):
        """Returns a new Frame object that provides accurate approximations to
        the transform returned by this frame. It is provided as a "hook" that
        can be invoked to speed up performance when the same frame must be
        evaluated repeatedly but within a very narrow range of times.

        Input:
            epoch       the single time value about which this frame will be
                        approximated.
            span        the range of times in seconds for which the approximated
                        frame will apply: [epoch - span, epoch + span].
            precision   if provided, a specific upper limit on the angular
                        precision of the new Frame.
            check       if True, then an attempt to evaluate the frame at a time
                        outside the allowed limits will raise a ValueError.
        """

        return self

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

        if oops.J2000FRAME == None:
            oops.J2000FRAME = J2000frame()

        oops.FRAME_REGISTRY = {"J2000": oops.J2000FRAME,
                               ("J2000","J2000"): oops.J2000FRAME}

    # This method is called below, as soon as class J2000frame() is defined.

    def register(self):
        """Registers a Frame definition. If the frame's ID is new, it is assumed
        to be the primary definition and is keyed by the ID alone. However, a
        primary definition must use a reference ID that is already registered.

        Otherwise or in addition, a secondary key is added to the registry if it
        is not already present:
            (frame_id, reference_id)
        This key also points to the same Frame object.
        """

        # Make sure the reference frame is registered; raise KeyError on failure
        reference = oops.FRAME_REGISTRY[self.reference_id]

        # If the ID is unregistered, insert this as a primary definition
        try:
            test = oops.FRAME_REGISTRY[self.frame_id]
        except KeyError:
            oops.FRAME_REGISTRY[self.frame_id] = self
            # Fill in the ancestry too
            self.ancestry = [self] + reference.ancestry

            # Also save the NullFrame
            oops.FRAME_REGISTRY[(self.frame_id,
                                 self.frame_id)] = NullFrame(self.frame_id)

        # If the tuple (self.frame_id, self.reference_id) is unregistered,
        # insert this as a secondary definition
        key = (self.frame_id, self.reference_id)
        try:
            test = oops.FRAME_REGISTRY[key]
        except KeyError:
            oops.FRAME_REGISTRY[key] = self

    def unregister(self):
        """Removes this frame from the registry."""

        # Note that we only delete the primary entry and any frame in which this
        # is one of the end points. If the frame is used as an intermediate step
        # between other frames, it will cease to be visible in the dictionary
        # but frames that use it will continue to function unchanged. However,
        # it is safest to remove all usage of a frame at the time it is
        # unregistered.

        frame_id = self.frame_id
        for key in oops.FRAME_REGISTRY.keys():
            if frame_id == key: del oops.FRAME_REGISTRY[key]

            if type(key) == type(()):
                if frame_id == key[0]: del oops.FRAME_REGISTRY[key]
                if frame_id == key[1]: del oops.FRAME_REGISTRY[key]

        oops.Path.unregister_frame(frame_id)

    def reregister(self):
        """Adds this frame to the registry, replacing any definition of the same
        name."""

        self.unregister()
        self.register()

    @staticmethod
    def lookup(key): return oops.FRAME_REGISTRY[key]

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
# Frame Generators
################################################################################

    @staticmethod
    def connect(target, reference):
        """Returns a Frame object that transforms from the given reference
        Frame to the given target Frame.

        Input:
            target      a Frame object or the registered ID of the destination
                        frame.
            reference   a Frame object or the registered ID of the starting
                        frame.

        The shape of the connected frame will will be the result of broadcasting
        the shapes of the target and reference.
        """

        # Convert to IDs
        target_id    = oops.as_frame_id(target)
        reference_id = oops.as_frame_id(reference)

        # If the path already exists, just return it
        try:
            return oops.FRAME_REGISTRY[(target_id, reference_id)]
        except KeyError: pass

        # Otherwise, construct it from the common ancestor...

        target_frame = oops.FRAME_REGISTRY[target_id]
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
        target    = oops.as_primary_frame(self)
        reference = oops.as_primary_frame(reference)

        # Check for compatibility
        if target.origin_id is not None and reference.origin_id is not None:
            assert target.origin_id == reference.origin_id

        # Find the common ancestry
        (target_ancestry,
         reference_ancestry) = Frame.common_ancestry(target, reference)

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
                target_frame = Frame.lookup((target.frame_id,
                                             target_ancestry[-1].reference_id))
            except KeyError:
                target_frame = LinkedFrame(target_ancestry)
                target_frame.register()

        # Look up or construct the common ancestor's frame from the reference
        if reference_ancestry == []:
            reference_frame = None
        elif len(reference_ancestry) == 1:
            reference_frame = reference
        else:
            try:
                reference_frame = Frame.lookup(
                                        (reference.frame_id,
                                         reference_ancestry[-1].reference_id))
            except KeyError:
                reference_frame = LinkedFrame(reference_ancestry)
                reference_frame.register()

        # Return the relative frame
        if reference_frame is None:
            if target_frame is None:
                result = NullFrame(target.frame_id)
            else:
                result = target_frame
        else:
            if target_frame is None:
                result = InverseFrame(reference_frame)
            else:
                result = RelativeFrame(target_frame, reference_frame)

        result.register()
        return result

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

#     @staticmethod
#     def str_ancestry(tuple):        # For debugging
#         list = ["(["]
# 
#         for item in tuple:
#             for frame in item:
#                 list += [frame.frame_id,"\", \""]
# 
#             list.pop()
#             list += ["\"], [\""]
# 
#         list.pop()
#         list += ["\"])"]
# 
#         return "".join(list)

########################################
# Arithmetic operators
########################################

    # unary "~" operator
    def __inverse__(self):
        return InverseFrame(self)

    # binary "*" operator
    def __mul__(self, arg):

        if oops.is_id(arg):
            return ProductFrame(self, arg)

        if arg.OOPS_CLASS == "Event":
            return self.rotate_event(arg)

        if arg.OOPS_CLASS == "Frame":
            return ProductFrame(self, arg)

        oops.raise_type_mismatch(self, "*", arg)

    # binary "/" operator
    def __div__(self, arg):

        if oops.is_id(arg):
            return RelativeFrame(self, arg)

        if arg.OOPS_CLASS == "Frame":
            return RelativeFrame(self, arg)

        oops.raise_type_mismatch(self, "/", arg)

    # string operations
    def __str__(self):
        return "Frame(" + self.frame_id + "/" + self.reference_id + ")"

################################################################################
# Required Subclasses
################################################################################

####################
# LinkedFrame
####################

class LinkedFrame(Frame):
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

        self.shape = oops.Array.broadcast_shape(tuple(self.frames))

    def transform_at_time(self, time):

        transform = self.frames[-1].transform_at_time(time)
        for frame in self.frames[-2::-1]:
           transform = frame.transform_at_time(time).rotate_transform(transform)

        return transform

####################
# ProductFrame
####################

class ProductFrame(Frame):
    """A Frame that returns the product of two frames, with the first
    multiplying the second."""

    def __init__(self, frame1, frame2):

        assert self.frame1.reference_id == self.frame2.frame_id

        self.frame1 = oops.as_primary_frame(frame1)
        self.frame2 = oops.as_primary_frame(frame2)

        # Required fields
        self.frame_id     = self.frame1.frame_id
        self.reference_id = self.frame2.reference_id

        # Identify the origin; confirm compatibility
        self.origin_id = frame1.origin_id
        if self.origin_id is None:
            self.origin_id = frame2.origin_id
        elif frame2.origin_id is not None:
            assert frame2.origin_id == self.origin_id

        self.shape = oops.Array.broadcast_shape((self.frame1, self.frame2))

    def transform_at_time(self, time):

        return self.frame1.transform_at_time(time).rotate_transform(
               self.frame2.transform_at_time(time))

####################
# RelativeFrame
####################

class RelativeFrame(Frame):
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

        self.shape = oops.Array.broadcast_shape((self.frame1, self.frame2))

    def transform_at_time(self, time):

        return self.frame1.transform_at_time(time).rotate_transform(
               self.frame2.transform_at_time(time).invert())

####################
# InverseFrame
####################

class InverseFrame(Frame):
    """A Frame that generates the inverse Transform of a given Frame.
    """

    def __init__(self, frame):

        self.oldframe = oops.as_frame(frame)

        # Required fields
        self.frame_id     = self.oldframe.reference_id
        self.reference_id = self.oldframe.frame_id
        self.origin_id    = self.oldframe.origin_id

        self.shape = self.oldframe.shape

    def transform_at_time(self, time):

        return self.oldframe.transform_at_time(time).invert()

####################
# NullFrame
####################

class NullFrame(Frame):
    """A Frame that always returns a null transform."""

    def __init__(self, frame_id, origin_id=None):

        # Required fields
        self.frame_id     = frame_id
        self.reference_id = frame_id
        self.origin_id    = None

        self.shape = []

    def transform_at_time(self, time):

        return oops.Transform.null_transform(self.frame_id)

    def __str__(self): return "Frame(" + self.frame_id + ")"

####################
# J2000frame
####################

class J2000frame(Frame):
    """The default inertial reference frame."""

    def __init__(self):

        self.ancestry = [self]
        self.frame_id = "J2000"
        self.reference_id = "J2000"
        self.origin_id = None

        self.shape = []

    def transform_at_time(self, time):

        return oops.Transform.null_transform(self)

    def __str__(self): return "Frame(" + self.frame_id + ")"

# Inialize the registry...
Frame.initialize_registry()

################################################################################
# UNIT TESTS
################################################################################

class Test_Frame(unittest.TestCase):

    def runTest(self):

        # Registry tests
        self.assertEquals(oops.FRAME_REGISTRY["J2000"], oops.J2000FRAME)

        # Extensive unit testing in SpiceFrame.py and SpicePath.py

        Frame.initialize_registry()

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
