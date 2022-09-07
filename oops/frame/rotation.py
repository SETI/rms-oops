################################################################################
# oops/frame/rotation.py: Subclass Rotation of class Frame
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from .           import Frame
from ..fittable  import Fittable
from ..transform import Transform

class Rotation(Frame, Fittable):
    """A Frame describing a fixed rotation about one axis of another frame."""

    PACKRAT_ARGS = ['angle', 'axis2', 'reference', 'frame_id']

    #===========================================================================
    def __init__(self, angle, axis, reference, id=None):
        """Constructor for a Rotation Frame.

        Input:
            angle       the angle of rotation in radians. Can be a Scalar
                        containing multiple values.
            axis        the rotation axis: 0 for x, 1 for y, 2 for z.
            reference   the frame relative to which this rotation is defined.
            id          the ID to use; None to leave the frame unregistered.
        """

        self.angle = Scalar.as_scalar(angle)
        self.shape = self.angle.shape

        self.axis2 = axis           # Most often, the Z-axis
        self.axis0 = (self.axis2 + 1) % 3
        self.axis1 = (self.axis2 + 2) % 3

        mat = np.zeros(self.shape + (3,3))
        mat[..., self.axis2, self.axis2] = 1.
        mat[..., self.axis0, self.axis0] = np.cos(self.angle.vals)
        mat[..., self.axis0, self.axis1] = np.sin(self.angle.vals)
        mat[..., self.axis1, self.axis1] =  mat[..., self.axis0, self.axis0]
        mat[..., self.axis1, self.axis0] = -mat[..., self.axis0, self.axis1]

        self.frame_id  = id
        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register()

        # We need a wayframe before we can create the transform
        self.transform = Transform(Matrix3(mat, self.angle.mask), Vector3.ZERO,
                                   self.wayframe, self.reference, self.origin)

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """Transform into this Frame at a Scalar of times."""

        return self.transform

    ############################################################################
    # Fittable interface
    ############################################################################

    def set_params(params):
        """Redefine the Fittable object, using this set of parameters.

        In this case, params is the set of angles of rotation.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        params = Scalar.as_scalar(params)
        assert params.shape == self.shape

        self.angle = params

        mat = np.zeros(self.shape + (3,3))
        mat[..., self.axis2, self.axis2] = 1.
        mat[..., self.axis0, self.axis0] = np.cos(self.angle.vals)
        mat[..., self.axis0, self.axis1] = np.sin(self.angle.vals)
        mat[..., self.axis1, self.axis1] =  mat[..., self.axis0, self.axis0]
        mat[..., self.axis1, self.axis0] = -mat[..., self.axis0, self.axis1]

        self.transform = Transform(Matrix3(mat, self.angle.mask), Vector3.ZERO,
                                   self.reference, self.origin)

    #===========================================================================
    def get_params(self):
        """The current set of parameters defining this fittable object.

        Return:         a Numpy 1-D array of floating-point numbers containing
                        the parameter values defining this object.
        """

        return self.angle.vals

    #===========================================================================
    def copy(self):
        """A deep copy of the given object.

        The copy can be safely modified without affecting the original.
        """

        return Rotation(self.angle.copy(), self.axis, self.reference_id)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Rotation(unittest.TestCase):

    def runTest(self):

        # Note: Unit testing is performed in surface/orbitplane.py

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
