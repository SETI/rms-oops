################################################################################
# oops_/frame/rotation.py: Subclass Rotation of class Frame
#
# 3/17/12 MRS - Created.
################################################################################

import numpy as np

from oops_.frame.frame_ import Frame
from oops_.array.all import *
from oops_.transform import Transform

import oops_.registry as registry

class Rotation(Frame):
    """Rotation is a Frame subclass describing a fixed rotation about one axis
    of another frame.
    """

    def __init__(self, angle, axis, reference, id=None):
        """Constructor for a Rotation Frame.

        Input:
            angle       the angle of rotation in radians. Can be a Scalar
                        containing multiple values.
            axis        the rotation axis: 0 for x, 1 for y, 2 for z.
            reference   the frame relative to which this rotation is defined.
            id          the ID to use; None to use a temporary ID.
        """

        self.angle = Scalar.as_scalar(angle)
        self.shape = self.angle.shape

        self.axis2 = axis           # Most often, the Z-axis
        self.axis0 = (self.axis2 + 1) % 3
        self.axis1 = (self.axis2 + 2) % 3

        mat = np.zeros(self.angle.shape + [3,3])
        mat[..., self.axis2, self.axis2] = 1.
        mat[..., self.axis0, self.axis0] = np.cos(angle)
        mat[..., self.axis1, self.axis1] = mat[..., self.axis0, self.axis0]
        mat[..., self.axis0, self.axis1] = np.sin(angle)
        mat[..., self.axis1, self.axis0] = -mat[..., self.axis0, self.axis1]

        self.frame_id = registry.as_frame_id(id)
        self.reference_id = registry.as_frame_id(reference)

        if id is None:
            self.frame_id = registry.temporary_frame_id()
        else:
            self.frame_id = id

        reference = registry.as_frame(self.reference_id)
        self.reference_id = registry.as_frame_id(reference)
        self.origin_id = reference.origin_id

        self.register()

        self.transform = Transform(Matrix3(mat, self.angle.mask), Vector3.ZERO,
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

class Test_Rotate(unittest.TestCase):

    def runTest(self):

        # Note: Unit testing is performed in surface/orbitplane.py

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
