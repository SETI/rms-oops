################################################################################
# Matrix3
#
# Modified 1/2/11 (MRS) -- Uses a cleaner style of imports.
# Modified 2/8/12 (MRS) -- Supports array masks; no unit tests added.
################################################################################

import numpy as np
import numpy.ma as ma

from baseclass import Array
from empty     import Empty
from scalar    import Scalar
from vector3   import Vector3

import utils as utils

class Matrix3(Array):
    """An arbitrary Array of 3x3 rotation matrices."""

    def __init__(self, arg, mask=False, units=None):

        if mask is not False: mask = np.asarray(mask)

        if units is not None:
            raise ValueError("a Matrix3 object cannot have units")

        if isinstance(arg, Matrix3):
            mask = arg.mask | mask
            arg = arg.vals

        elif isinstance(arg, ma.MaskedArray):
            if arg.mask != ma.nomask:
                mask = mask | np.any(np.any(arg.mask, axis=-1), axis=-1)
            arg = arg.data

        self.vals = np.asfarray(arg)
        ashape = list(self.vals.shape)

        self.rank  = 2
        self.item  = ashape[-2:]
        self.shape = ashape[:-2]
        self.mask  = mask
        self.units = None

        if self.item != [3,3]:
            raise ValueError("shape of a Matrix3 array must be [...,3,3]")

        if (self.mask is not False) and (list(self.mask.shape) != self.shape):
            raise ValueError("mask array is incompatible with Matrix3 shape")

        return

    @staticmethod
    def as_matrix3(arg):
        if isinstance(arg, Matrix3): return arg
        return Matrix3(arg)

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Matrix3): arg = Matrix3(arg)
        return arg

    def rotate(self, arg):
        """Matrix3 multiplied by a Vector3."""

        if isinstance(arg, Empty): return arg
        arg = Vector3.as_vector3(arg)
        return Vector3(utils.mxv(self.vals, arg.vals), self.mask | arg.mask,
                       arg.units)

    def unrotate(self, arg):
        """Matrix3 inverse multiplied by a Vector3."""

        if isinstance(arg, Empty): return arg
        arg = Vector3.as_vector3(arg)
        return Vector3(utils.mtxv(self.vals, arg.vals), self.mask | arg.mask,
                       arg.units)

    def rotate_matrix(self, arg):
        """Matrix3 multiplied by another Matrix3."""

        if isinstance(arg, Empty): return arg
        arg = Matrix3.as_matrix3(arg)
        return Matrix3(utils.mxm(self.vals, arg.vals), self.mask | arg.mask)

    def rotate_inverse_matrix(self, arg):
        """Matrix3 multiplied by the inverse of another Matrix3."""

        if isinstance(arg, Empty): return arg
        arg = Matrix3.as_matrix3(arg)
        return Matrix3(utils.mxmt(self.vals, arg.vals), self.mask | arg.mask)

    def unrotate_matrix(self, arg):
        """Matrix3 inverse multiplied by another Matrix3."""

        if isinstance(arg, Empty): return arg
        arg = Matrix3.as_matrix3(arg)
        return Matrix3(utils.mtxm(self.vals, arg.vals), self.mask | arg.mask)

    def invert(self):
        """Inverse rotation matrix."""

        return Matrix3(self.vals.swapaxes(-2,-1), self.mask)

    def axis(self, axis):
        """Returns one of the destination coordinate frame's axes in the frame
        of the origin. These are equivalent to matrix_T*(1,0,0) for axis == 0;
        matrix_T*(0,1,0) for axis == 1; matrix_T*(0,0,1) for axis == 2."""

        return Vector3(self.vals[..., axis], self.mask)

    @staticmethod
    def twovec(v1, axis1, v2, axis2):
        """Returns the rotation matrix to a coordinate frame having the first
        vector along a specified axis and the second vector in a specified
        half-plane.

        axis1 and axis2 are 0 for X, 1 for Y and 2 for Z."""

        v1vals = np.asfarray(Vector3(v1).vals)
        v2vals = np.asfarray(Vector3(v2).vals)
        return Matrix3(utils.twovec(v1vals, axis1, v2vals, axis2))

    ####################################################
    # Overrides of binary arithmetic operators
    ####################################################

    # Matrix3 (*) operator
    def __mul__(self, arg):

        # Matrix3 * Matrix3 is direct matrix multiply
        try:
            return self.rotate_matrix(arg)
        except: pass

        # Anything else is coordinate rotation
        return self.rotate(arg)

    # Matrix3 (/) operator
    def __div__(self, arg):
        if Array.is_empty(arg): return arg

        # First Matrix3 times inverse of second
        return self.rotate_inverse_matrix(arg)

    # Matrix3 (*=) operator
    def __imul__(self, arg):

        result = self.rotate_matrix(arg)
        self.vals = result.vals
        self.mask = result.mask
        return self

    # (/=) operator
    def __idiv__(self, arg):

        result = self.rotate_inverse_matrix(arg)
        self.vals = result.vals
        self.mask = result.mask
        return self

    def __add__(self, arg): Array.raise_type_mismatch(self, "+", arg)
    def __sub__(self, arg): Array.raise_type_mismatch(self, "-", arg)
    def __mod__(self, arg): Array.raise_type_mismatch(self, "%", arg)

    def __iadd__(self, arg): Array.raise_type_mismatch(self, "+=", arg)
    def __isub__(self, arg): Array.raise_type_mismatch(self, "-=", arg)
    def __imod__(self, arg): Array.raise_type_mismatch(self, "%=", arg)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Matrix3(unittest.TestCase):

    def runTest(self):

        eps = 1.e-15

        a = Matrix3(np.random.rand(2,1,4,3,3))
        b = Matrix3(np.random.rand(  3,4,3,3))
        v = Vector3(np.random.rand(1,3,1,3))

        axb  = a.rotate_matrix(b)
        test = a.rotate_matrix(b.vals)
        self.assertTrue(np.all(axb.vals - test.vals > -eps))
        self.assertTrue(np.all(axb.vals - test.vals <  eps))

        test = a * b
        self.assertTrue(np.all(axb.vals - test.vals > -eps))
        self.assertTrue(np.all(axb.vals - test.vals <  eps))

        atxb = a.unrotate_matrix(b)
        test = Matrix3(utils.mtxm(a.vals, b.vals))
        self.assertTrue(np.all(atxb.vals - test.vals > -eps))
        self.assertTrue(np.all(atxb.vals - test.vals <  eps))

        axbt = a.rotate_matrix(b.invert())
        test = a / b
        self.assertTrue(np.all(axbt.vals - test.vals > -eps))
        self.assertTrue(np.all(axbt.vals - test.vals <  eps))

        for i in range(2):
          for j in range(3):
            for k in range(4):
                am = np.matrix(a.vals[i,0,k])
                vm = np.matrix(v.vals[0,j,0].reshape((3,1)))

                axv = a[i,0,k].rotate(v[0,j,0]).vals[..., np.newaxis]
                self.assertTrue(np.all(am*vm - axv > -eps))
                self.assertTrue(np.all(am*vm - axv <  eps))

                axv = (a[i,0,k] * v[0,j,0]).vals[..., np.newaxis]
                self.assertTrue(np.all(am*vm - axv > -eps))
                self.assertTrue(np.all(am*vm - axv <  eps))

                atxv = a[i,0,k].unrotate(v[0,j,0]).vals[..., np.newaxis]
                self.assertTrue(np.all(am.T*vm - atxv > -eps))
                self.assertTrue(np.all(am.T*vm - atxv <  eps))

        a = Matrix3(np.random.rand(2,3,4,3,3))
        b = Matrix3(np.random.rand(  3,1,3,3))

        test = a.copy()
        test *= b
        self.assertTrue(np.all(test.vals - (a*b).vals > -eps))
        self.assertTrue(np.all(test.vals - (a*b).vals <  eps))

        test = a.copy()
        test /= b
        self.assertTrue(np.all(test.vals - (a/b).vals > -eps))
        self.assertTrue(np.all(test.vals - (a/b).vals <  eps))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
