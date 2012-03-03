################################################################################
# oops_/array_/matrix3.py: Matrix3 subclass of class Array
#
# Modified 1/2/11 (MRS) -- Uses a cleaner style of imports.
# Modified 2/8/12 (MRS) -- Supports array masks; no unit tests added.
# Modified 2/25/12 (MRS) -- Made into a subclass of MatrixN.
################################################################################

import numpy as np
import numpy.ma as ma

from baseclass import Array
from empty     import Empty
from scalar    import Scalar
from vector3   import Vector3

import utils

class Matrix3(MatrixN):
    """An arbitrary Array of 3x3 rotation matrices."""

    def __init__(self, arg, mask=False, units=None):

        return Array.__init__(self, arg, mask, units, 2, item=[3,3],
                                    float=True, dimensionless=True)

    @staticmethod
    def as_matrix3(arg):
        if not isinstance(arg, Matrix3): arg = Matrix3(arg)
        return arg

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Matrix3): arg = Matrix3(arg)
        return arg

    def multiply_matrix(self, arg):
        """A general definition of matrix * matrix."""

        if isinstance(arg, Matrix3):
            return self.rotate_matrix3(arg)
        else:
            return MatrixN.multiply_matrix(self, arg)

    def multiply_vector(self, arg):
        """A general definition of matrix * vector."""

        if isinstance(arg, Vecrtor3):
            return self.rotate_vector3(arg)
        else:
            return MatrixN.multiply_vector(self, arg)

    def inverse(self):
        """A general definition of matrix inverse."""

        return self.transpose()

    def transpose(self):
        """Transpose rotation matrix."""

        return Matrix3(self.vals.swapaxes(-2,-1), self.mask)

    def T(self):
        """Transpose rotation matrix."""

        return Matrix3(self.vals.swapaxes(-2,-1), self.mask)

    ############################

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

    ############################

    def rotate(self, arg):
        """Matrix3  rotation of anything. Note that rotation of a scalar returns
        the same scalar."""

        if isinstance(arg, Array):
            if rank == 0:
                return arg

            if rank == 1:
                vals1 = self.vals
                vals2 = arg.vals[..., np.newaxis, :]
                return Vector3(np.sum(vals1*vals2, axis=-1),
                               self.mask | arg.mask)

            if rank == 2:
                vals1 = self.vals[..., np.newaxis, :]
                vals2 = arg.vals[..., np.newaxis, :, :].swapaxes(-1,-2)
                return Matrix3(np.sum(vals1*vals2, axis=-1),
                               self.mask | arg.mask)

        if isinstance(arg, np.ndarray):
            if len(arg.shape) >= 2 and arg.shape[-2:] = (3,3):
                vals1 = self.vals[..., np.newaxis, :]
                vals2 = arg[..., np.newaxis, :, :].swapaxes(-1,-2)
                return Matrix3(np.sum(vals1*vals2, axis=-1), self.mask)

            elif len(arg.shape) >= 1 and arg.shape[-1:] = (3,):
                vals1 = self.vals[..., np.newaxis, :]
                vals2 = arg[..., np.newaxis, :]
                return Vector3(np.sum(vals1*vals2, axis=-1), self.mask)

            else:
                return Scalar(arg)

        else:
            return Scalar(arg)

    ############################

    def unrotate(self, arg):
        """Matrix3 inverse rotation of anything.Note that rotation of a scalar
        returns the same scalar."""

        if isinstance(arg, Array):
            if rank == 0:
                return arg

            if rank == 1:
                vals1 = self.vals
                vals2 = arg.vals[..., np.newaxis]
                return Vector3(np.sum(vals1*vals2, axis=-2),
                               self.mask | arg.mask)

            if rank == 2:
                vals1 = self.vals[..., np.newaxis, :]
                vals2 = arg.vals[..., np.newaxis, :, :]
                return Matrix3(np.sum(vals1*vals2, axis=-2),
                               self.mask | arg.mask)

        if isinstance(arg, np.ndarray):
            if len(arg.shape) >= 2 and arg.shape[-2:] = (3,3):
                vals1 = self.vals[..., np.newaxis, :]
                vals2 = arg[..., np.newaxis, :, :]
                return Matrix3(np.sum(vals1*vals2, axis=-2), self.mask)

            elif len(arg.shape) >= 1 and arg.shape[-1:] = (3,):
                vals1 = self.vals
                vals2 = arg[..., np.newaxis]
                return Vector3(np.sum(vals1*vals2, axis=-2), self.mask)

            else:
                return Scalar(arg)

        else:
            return Scalar(arg)

    ####################################################
    # Overrides of multiplication operators
    ####################################################

    def __mul__(self, arg):
        result = self.rotate(arg)
        if result is arg: return result

        # Multiply subarrays if necessary
        if Array.SUBARRAY_ARITHMETIC:
            result.mul_subarrays(self, arg)

        return result

    def __rmul__(self, arg):

        # Handles MatrixN * Matrix3 and VectorN * Matrix3
        if isinstance(arg, Array):
            if arg.rank > 0:
                return arg * self

        # Handles Numpy array * Matrix
        elif isinstance(arg, np.ndarray):
            return Matrix3(arg) * self

        return NotImplemented

    def __imul__(self, arg):

        result = self.rotate_matrix(arg)
        self.vals[...] = result.vals[...]
        self.mask |= result.mask

        if Array.SUBARRAY_ARITHMETIC:
            self.imul_subarrays(arg)

        return self

    ####################################################
    # Overrides of division operators
    ####################################################

    def __div__(self, arg):
        raise ValueError("Matrix3 division is not supported")

    def __rdiv__(self, arg):
        raise ValueError("Matrix3 division is not supported")

    def __idiv__(self, arg):
        raise ValueError("Matrix3 division is not supported")

    def __invert__(self):
        return self.transpose()

    ####################################################
    # Overrides of other arithmetic operators
    ####################################################

    # Add and subtract are useful for testing so they are not overridden
    # def __add__(self, arg): Array.raise_type_mismatch(self, "+", arg)
    # def __sub__(self, arg): Array.raise_type_mismatch(self, "-", arg)
    def __mod__(self, arg): Array.raise_type_mismatch(self, "%", arg)

    def __iadd__(self, arg): Array.raise_type_mismatch(self, "+=", arg)
    def __isub__(self, arg): Array.raise_type_mismatch(self, "-=", arg)
    def __imod__(self, arg): Array.raise_type_mismatch(self, "%=", arg)

    # abs is useful to compare the difference of two matrices
    def __abs__(self):
        return Scalar(np.sqrt(np.sum(np.sum(self.vals**2, axis=-1), axis=-1)))

################################################################################
# Once defined, register with Array class
################################################################################

Array.MATRIX3_CLASS = Matrix3

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
