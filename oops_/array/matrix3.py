################################################################################
# oops_/array_/matrix3.py: Matrix3 subclass of class Array
#
# Modified 1/2/11 (MRS) -- Uses a cleaner style of imports.
# Modified 2/8/12 (MRS) -- Supports array masks; no unit tests added.
# Modified 2/25/12 (MRS) -- Made into a subclass of MatrixN.
# 3/2/12 MRS: Better integrated with VectorN and MatrixN.
################################################################################

import numpy as np
import numpy.ma as ma

from array_  import Array
from empty   import Empty
from scalar  import Scalar
from vector3 import Vector3
from matrixn import MatrixN

import utils

class Matrix3(Array):
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

    def invert(self):
        """A general definition of matrix inverse."""

        return self.transpose()

    def transpose(self):
        """Transpose rotation matrix."""

        return Matrix3(self.vals.swapaxes(-2,-1), self.mask)

    def T(self):
        """Transpose rotation matrix."""

        return Matrix3(self.vals.swapaxes(-2,-1), self.mask)

    ############################

    def column(self, axis):
        """Returns one column of the matrix as a Vector3."""
        return Vector3(self.vals[..., axis], self.mask)

    def row(self, axis):
        """Returns one row of the matrix as a Vector3."""
        return Vector3(self.vals[..., axis, :], self.mask)

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

            # Rotation of a scalar leaves it unchanged
            if arg.rank == 0:
                return arg

            # Rotation of a vector
            if arg.rank == 1:
                return self.rotate_vector3(arg)

            if arg.rank == 2:
                return self.rotate_matrixn(arg)

        # Not recommended but can be handled
        arg = np.array(arg)
        if len(arg.shape) >= 2 and arg.shape[-2:] == (3,3):
            return self.rotate_matrix3(arg)

        elif len(arg.shape) >= 1 and arg.shape[-1:] == (3,):
            return self.rotate_vector3(arg)

        else:
            return Scalar(arg)

    ############################

    def unrotate(self, arg):
        """Matrix3 inverse rotation of anything. Note that rotation of a scalar
        returns the same scalar."""

        if isinstance(arg, Array):
            if arg.rank == 0:
                return arg

            if arg.rank == 1:
                return self.unrotate_vector3(arg)

            if arg.rank == 2:
                return self.rotate_matrixn(arg)

        arg = np.array(arg)
        if len(arg.shape) >= 2 and arg.shape[-2:] == (3,3):
            return self.rotate_matrix3(arg)

        elif len(arg.shape) >= 1 and arg.shape[-1:] == (3,):
            return self.unrotate_vector3(arg)

        else:
            return Scalar(arg)

    ############################

    def rotate_vector3(self, arg):
        arg = Vector3.as_vector3(arg)
        arg_vals = arg.vals[..., np.newaxis, :]
        return Vector3(np.sum(self.vals * arg_vals, axis=-1),
                              self.mask | arg.mask, arg.units)

    def unrotate_vector3(self, arg):
        arg = Vector3.as_vector3(arg)
        arg_vals = arg.vals[..., np.newaxis]
        return Vector3(np.sum(self.vals * arg_vals, axis=-2),
                              self.mask | arg.mask, arg.units)

    def rotate_matrix3(self, arg):
        arg = Matrix3.as_matrix3(arg)
        vals1 = self.vals[..., np.newaxis, :]
        vals2 = arg.vals[..., np.newaxis, :, :].swapaxes(-1,-2)
        return Matrix3(np.sum(vals1 * vals2, axis=-1), self.mask | arg.mask)

    def unrotate_matrix3(self, arg):
        arg = Matrix3.as_matrix3(arg)
        vals1 = self.vals[..., np.newaxis]
        vals2 = arg.vals[..., np.newaxis, :]
        return Matrix3(np.sum(vals1 * vals2, axis=-3), self.mask | arg.mask)

    def rotate_matrixn(self, arg):
        """Matrix multiply in which it returns class Matrix3 if it can;
        otherwise it returns MatrixN. Multiplication by a MatrixN with item
        shape [1,N] is treated as scalar multiply."""

        if isinstance(arg, Array):
            if arg.item[0] == 1: return arg     # This is scalar multiply
            use_matrix3 = arg.item[-1] == 3
        else:
            use_matrix3 = arg.shape[-1] == 3

        vals1 = self.vals[..., np.newaxis, :]
        vals2 = arg.vals[..., np.newaxis, :, :].swapaxes(-1,-2)

        if use_matrix3:
            return Matrix3(np.sum(vals1 * vals2, axis=-1), self.mask | arg.mask)
        else:
            return MatrixN(np.sum(vals1 * vals2, axis=-1), self.mask | arg.mask)

    def unrotate_matrixn(self, arg):
        """Matrix inverse multiply in which it returns class Matrix3 if it can;
        otherwise it returns MatrixN. Multiplication by a MatrixN with item
        shape [1,N] is treated as scalar multiply."""

        if isinstance(arg, Array):
            if arg.item[0] == 1: return arg     # This is scalar multiply
            use_matrix3 = arg.item[-1] == 3
        else:
            use_matrix3 = arg.shape[-1] == 3

        vals1 = self.vals[..., np.newaxis, :]
        vals2 = arg.vals[..., np.newaxis, :, :]

        if use_matrix3:
            return Matrix3(np.sum(vals1 * vals2, axis=-2), self.mask | arg.mask)
        else:
            return MatrixN(np.sum(vals1 * vals2, axis=-2), self.mask | arg.mask)

    ####################################################
    # Overrides of multiplication operators
    ####################################################

    def __mul__(self, arg):
        result = self.rotate(arg)
        if result is arg: return result

        # Multiply subfields if necessary
        if not Array.IGNORE_SUBFIELDS:
            result.mul_subfields(self, arg)

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

        result = self.rotate_matrix3(arg)
        self.vals[...] = result.vals[...]
        self.mask |= result.mask

        if not Array.IGNORE_SUBFIELDS:
            self.imul_subfields(arg)

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

        axb  = a.rotate_matrix3(b)
        test = a.rotate_matrix3(b.vals)
        self.assertTrue(np.all(axb.vals - test.vals > -eps))
        self.assertTrue(np.all(axb.vals - test.vals <  eps))

        test = a * b
        self.assertTrue(np.all(axb.vals - test.vals > -eps))
        self.assertTrue(np.all(axb.vals - test.vals <  eps))

        atxb = a.unrotate_matrix3(b)
        test = Matrix3(utils.mtxm(a.vals, b.vals))
        self.assertTrue(np.all(atxb.vals - test.vals > -eps))
        self.assertTrue(np.all(atxb.vals - test.vals <  eps))

        axbt = a.rotate_matrix3(b.invert())
#         test = a / b
#         self.assertTrue(np.all(axbt.vals - test.vals > -eps))
#         self.assertTrue(np.all(axbt.vals - test.vals <  eps))

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

#         test = a.copy()
#         test /= b
#         self.assertTrue(np.all(test.vals - (a/b).vals > -eps))
#         self.assertTrue(np.all(test.vals - (a/b).vals <  eps))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
