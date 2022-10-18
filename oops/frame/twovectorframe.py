################################################################################
# oops/frame/twovectorframe.py: Subclass TwoVectorFrame of class Frame
################################################################################

from polymath import Qube, Vector3, Matrix3

from .           import Frame
from ..transform import Transform

class TwoVectorFrame(Frame):
    """A Frame subclass describing a frame that is fixed relative to another
    frame.

    It is described by two vectors. The first vector is one axis of the frame
    and the second vector points in the half-plane of another axis.
    """

    XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2}

    FRAME_IDS = {}  # frame_id to use if a frame already exists upon un-pickling

    #===========================================================================
    def __init__(self, frame, vector1, axis1, vector2, axis2, frame_id='+',
                       unpickled=False):
        """Constructor for a TwoVectorFrame.

        Input:
            frame       the frame relative to which this frame is defined.

            vector1     vector describing an axis.

            axis1       'X', 'Y', or 'Z', indicating the axis defined by the
                        first vector..

            vector2     a vector which, along with vector1, defines the half
                        plane in which a second axis falls.

            axis2       'X', 'Y', or 'Z', indicating the axis defined by the
                        second vector.

            frame_id    the ID under which the frame will be registered. None to
                        leave the frame unregistered. If the value begins with
                        "+", then the "+" is replaced by an underscore and the
                        result is appended to the name of the reference frame.
                        If the name is "+" alone, then the registered name is
                        that of the reference frame appended with '_TWOVECTOR'.

            unpickled   True if this frame has been read from a pickle file.
        """

        self.vector1 = Vector3.as_vector3(vector1)
        self.vector2 = Vector3.as_vector3(vector2)
        self.axis1 = axis1
        self.axis2 = axis2

        assert (self.axis1 in 'XYZ')
        assert (self.axis2 in 'XYZ')

        self.reference = Frame.as_wayframe(frame)

        self.shape = Qube.broadcasted_shape(self.vector1, self.vector2,
                                            self.reference)
        self.keys = set()

        self.origin = self.reference.origin

        # Fill in the frame ID
        self._state_frame_id = frame_id
        if frame_id is None:
            self.frame_id = Frame.temporary_frame_id()
        elif frame_id.startswith('+') and len(frame_id) > 1:
            self.frame_id = self.reference.frame_id + '_' + frame_id[1:]
        elif frame_id == '+':
            self.frame_id = self.reference.frame_id + '_TWOVECTOR'
        else:
            self.frame_id = frame_id

        # Register if necessary
        self.register(unpickled=unpickled)

        # Derive the tranform now
        matrix = Matrix3.twovec(self.vector1, TwoVectorFrame.XYZDICT[axis1],
                                self.vector2, TwoVectorFrame.XYZDICT[axis2])

        self.transform = Transform(matrix, Vector3.ZERO,
                                   self.wayframe, self.reference)

        z_axis = matrix.row_vector(2, Vector3)
        self.node = Vector3.ZAXIS.ucross(z_axis)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.frame_id in Frame.WAYFRAME_REGISTRY):
                key = (self.reference.frame_id,
                       tuple(self.vector1.vals), self.axis1,
                       tuple(self.vector2.vals), self.axis2)
                TwoVectorFrame.FRAME_IDS[key] = self.frame_id

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.reference, self.vector1, self.axis1,
                                self.vector2, self.axis2, self.shape)

    def __setstate__(self, state):
        # If this frame matches a pre-existing frame, re-use its ID
        (frame, vector1, axis1, vector2, axis2, shape) = state
        if self.shape == ():
            key = (frame.frame_id, tuple(vector1.vals), axis1,
                                   tuple(vector2.vals), axis2)
            frame_id = TwoVectorFrame.FRAME_IDS.get(key, None)
        else:
            frame_id = None

        self.__init__(frame, vector1, axis1, vector2, axis2, frame_id=frame_id,
                      unpickled=True)

    #===========================================================================
    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

        return self.transform

    #===========================================================================
    def node_at_time(self, time, quick={}):

        return self.node

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_TwoVectorFrame(unittest.TestCase):

    def runTest(self):

        pass    # TBD

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
