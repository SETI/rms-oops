################################################################################
# oops/frame_/postarg.py: Subclass PosTarg of class Frame
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.transform    import Transform

class PosTarg(Frame):
    """PosTarg is a Frame subclass describing a fixed rotation about the X and
    Y axes, so that the Z-axis of another frame falls at a slightly different
    position in this frame.
    """

    def __init__(self, xpos, ypos, reference, id=None):
        """Constructor for a PosTarg Frame.

        Input:
            xpos        the X-position of the reference frame's Z-axis in this
                        frame, in radians.
            ypos        the Y-position of the reference frame's Z-axis in this
                        frame, in radians.
            reference   the frame relative to which this frame is defined.
            id          the ID to use; None to leave the frame unregistered.
        """

        self.xpos = float(xpos)
        self.ypos = float(ypos)

        cos_x = np.cos(self.xpos)
        sin_x = np.sin(self.xpos)

        cos_y = np.cos(self.ypos)
        sin_y = np.sin(self.ypos)

        xmat = Matrix3([[1.,  0.,    0.   ],
                        [0.,  cos_y, sin_y],
                        [0., -sin_y, cos_y]])

        ymat = Matrix3([[ cos_x, 0., sin_x],
                        [ 0.,    1., 0.   ],
                        [-sin_x, 0., cos_x]])

        mat = ymat * xmat

        self.frame_id  = id
        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.shape     = self.reference.shape
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register()

        # It needs a wayframe before we can define the transform
        self.transform = Transform(mat, Vector3.ZERO,
                                   self.frame_id, self.reference, self.origin)

    ########################################

    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

        return self.transform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_PosTarg(unittest.TestCase):

    def runTest(self):

        Frame.reset_registry()

        postarg = PosTarg(0.0001, 0.0002, "J2000")
        transform = postarg.transform_at_time(0.)
        rotated = transform.rotate(Vector3.ZAXIS)

        self.assertTrue(abs(rotated.vals[0] - 0.0001) < 1.e-8)
        self.assertTrue(abs(rotated.vals[1] - 0.0002) < 1.e-8)

        Frame.reset_registry()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
