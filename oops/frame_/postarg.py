################################################################################
# oops/frame_/postarg.py: Subclass PosTarg of class Frame
#
# 8/4/12 MRS - Created.
################################################################################

import numpy as np

from oops.frame_.frame import Frame
from oops.array_       import *
from oops.transform    import Transform

import oops.registry as registry

class PosTarg(Frame):
    """PosTarg is a Frame subclass describing a fixed rotation about the X and
    Y axes, so that the Z-axis of another frame falls at a slightly different
    position in this frame.
    """

    def __init__(self, xpos, ypos, reference, id=None):
        """Constructor for a Rotation Frame.

        Input:
            xpos        the X-position of the reference frame's Z-axis in this
                        frame, in radians.
            ypos        the Y-position of the reference frame's Z-axis in this
                        frame, in radians.
            reference   the frame relative to which this frame is defined.
            id          the ID to use; None to use a temporary ID.
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

        self.frame_id = registry.as_frame_id(id)
        self.reference_id = registry.as_frame_id(reference)

        if id is None:
            self.frame_id = registry.temporary_frame_id()
        else:
            self.frame_id = id

        reference = registry.as_frame(self.reference_id)
        self.reference_id = registry.as_frame_id(reference)
        self.origin_id = reference.origin_id

        self.reregister()

        self.transform = Transform(mat, Vector3.ZERO,
                                   self.reference_id, self.origin_id)

########################################

    def transform_at_time(self, time, quick=False):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        return self.transform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_PosTarg(unittest.TestCase):

    def runTest(self):

        import oops

        registry.initialize()

        postarg = PosTarg(0.0001, 0.0002, "J2000")
        transform = postarg.transform_at_time(0.)
        rotated = transform.rotate(Vector3.ZAXIS)

        self.assertTrue(abs(rotated.vals[0] - 0.0001) < 1.e-8)
        self.assertTrue(abs(rotated.vals[1] - 0.0002) < 1.e-8)

        registry.initialize()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
