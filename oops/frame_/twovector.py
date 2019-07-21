################################################################################
# oops/frame_/twovector.py: Subclass TwoVector of class Frame
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.transform    import Transform

class TwoVector(Frame):
    """TwoVector is a Frame subclass describing a frame that is fixed relative
    to another frame. It is described by two vectors. The first vector is one
    axis of the frame and the second vector points in the half-plane of another
    axis.
    """

    PACKRAT_ARGS = ['reference', 'vector1', 'axis1', 'vector2', 'axis2', ]

    XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2}

    def __init__(self, frame, vector1, axis1, vector2, axis2,
                       id='+'):
        """Constructor for a RingFrame Frame.

        Input:
            frame       the frame relative to which this frame is defined.

            vector1     vector describing an axis.

            axis1       'X', 'Y', or 'Z', indicating the axis defined by the
                        first vector..

            vector2     a vector which, along with vector1, defines the half
                        plane in which a second axis falls.

            axis2       'X', 'Y', or 'Z', indicating the axis defined by the
                        second vector.

            id          the ID under which the frame will be registered. None to
                        leave the frame unregistered. If the value begins with
                        "+", then the "+" is replaced by an underscore and the
                        result is appended to the name of the reference frame.
                        If the name is "+" alone, then the registered name is
                        that of the reference frame appended with '_TWOVECTOR'.
        """

        self.vector1 = Vector3.as_vector3(vector1)
        self.vector2 = Vector3.as_vector3(vector2)
        self.axis1 = axis1
        self.axis2 = axis2

        self.reference = Frame.as_wayframe(frame)

        self.shape = Qube.broadcasted_shape(self.vector1, self.vector2)
        self.keys = set()

        self.origin = self.reference.origin

        # Fill in the frame ID
        if id is None:
            self.frame_id = Frame.temporary_frame_id()
        elif id.startswith('+') and len(id) > 1:
            self.frame_id = self.reference.frame_id + '_' + id[1:]
        elif id == '+':
            self.frame_id = self.reference.frame_id + '_TWOVECTOR'
        else:
            self.frame_id = id

        # Register if necessary
        if id:
            self.register()
        else:
            self.wayframe = self

        # Derive the tranform now
        matrix = Matrix3.twovec(self.vector1, TwoVector.XYZDICT[axis1],
                                self.vector2, TwoVector.XYZDICT[axis2])

        self.transform = Transform(matrix, Vector3.ZERO,
                                   self.wayframe, self.reference)

        z_axis = matrix.row_vector(2, Vector3)
        self.node = Vector3.ZAXIS.ucross(z_axis)

    ########################################

    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

        return self.transform

    ########################################

    def node_at_time(self, time, quick={}):

        return self.node

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_TwoVector(unittest.TestCase):

    def runTest(self):

        pass    # TBD

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
