################################################################################
# oops_/array_/matrixn.py: MatrixN subclass of class Array
#
# 3/2/12 Created (MRS)
################################################################################

import numpy as np
import numpy.ma as ma

from oops_.array.array_  import Array
from oops_.array.pair    import Pair
from oops_.array.scalar  import Scalar
from oops_.array.vector3 import Vector3
from oops_.array.vectorn import VectorN

class MatrixN(Array):
    """An Array of arbitrary matrices."""

    def __init__(self, arg, mask=False, units=None):
        return Array.__init__(self, arg, mask, units, 2, item=None,
                                    float=True, dimensionless=True)

    @staticmethod
    def as_matrixn(arg):
        if isinstance(arg, MatrixN): return arg
        return MatrixN(arg)

    @staticmethod
    def as_standard(arg):
        if isinstance(arg, MatrixN): return arg
        return MatrixN(arg)

    def as_vectorn(self):
        """Converts a 1xN or Nx1 MatrixN to a VectorN."""

        if self.item[1] == 1:
            return VectorN(self.vals[...,0], self.mask)

        if self.item[0] == 1:
            return VectorN(self.vals[...,0,:], self.mask)

        raise ValueError("MatrixN with item shape " + str(self.item) +
                         " cannot be converted to a vector")

    def as_vector3(self):
        """Converts a 1xN or Nx1 MatrixN to a Vector3."""

        if self.item[1] == 1:
            return Vector3(self.vals[...,0], self.mask)

        if self.item[0] == 1:
            return Vector3(self.vals[...,0,:], self.mask)

        raise ValueError("MatrixN with item shape " + str(self.item) +
                         " cannot be converted to a Vector3")

    def as_scalar(self):
        """Converts a 1x1 MatrixN to a Scalar."""

        if self.item[-2:] == [1,1]:
            return Scalar(self.vals[...,0,0], self.mask)

        raise ValueError("MatrixN with item shape " + str(self.item) +
                         " cannot be converted to a Scalar")

    def as_row(self, row):
        """Returns the selected row of an M x N MatrixN as a VectorN of item
        shape [N]."""

        return VectorN(self.vals[...,row,:], self.mask)

    def as_rows(self):
        """Converts an M x N MatrixN to a list containing M row matrices as
        VectorN objects of item shape [N]."""

        list = []
        for row in range(self.item[0]):
            list.append(VectorN(self.vals[...,row,:], self.mask))

        return list

    def as_column(self, col):
        """Returns the selected column of an M x N MatrixN as a VectorN of item
        shape [M]."""

        return VectorN(self.vals[..., col], self.mask)

    def as_columns(self):
        """Converts an M x N MatrixN to a list containing N column matices as
        VectorN objects of item shape [M]."""

        list = []
        for col in range(self.item[1]):
            list.append(MatrixN(self.vals[...,col], self.mask))

        return list

    def multiply_matrix(self, arg):
        """A general definition of matrix * matrix."""

        if (self.item[1] != arg.item[0]):
            raise ValueError("shape mismatch for matrix multiply: " +
                             str(self.item) + " * " + str(arg.item))
        vals1 = self.vals[..., np.newaxis, :]
        vals2 = arg.vals[..., np.newaxis, :, :].swapaxes(-1,-2)

        return MatrixN(np.sum(vals1*vals2, axis=-1), self.mask | arg.mask)

    def multiply_vector(self, arg):
        """A general definition of matrix * vector."""

        if self.item[1] != arg.item[0]:
            raise ValueError("shape mismatch for matrix multiply " + 
                             str(self.item) + " * " + str(arg.item))
        vals1 = self.vals
        vals2 = arg.vals[..., np.newaxis,:]

        return VectorN(np.sum(vals1*vals2, axis=-1), self.mask | arg.mask,
                                                     arg.units)

    def inverse(self):
        """A general definition of matrix inverse."""

        if self.item[0] != self.item[1]:
            raise ValueError("only square matrices can be inverted: shape is " +
                             str(self.item))

        # 2 x 2 case
        if self.item[0] == 2:
            inverse_vals = np.empty(self.shape + [2,2])

            inverse_vals[...,0,0] =  self.vals[...,1,1]
            inverse_vals[...,0,1] = -self.vals[...,0,1]
            inverse_vals[...,1,0] = -self.vals[...,1,0]
            inverse_vals[...,1,1] =  self.vals[...,0,0]

            det = (self.vals[...,0,0] * self.vals[...,1,1] -
                   self.vals[...,0,1] * self.vals[...,1,0])

            return MatrixN(inverse_vals, self.mask) / det

        # Remainder are TBD
        return NotImplemented

    def transpose(self):
        """Transpose of matrix."""

        return MatrixN(self.vals.swapaxes(-2,-1), self.mask)

    def T(self):
        """Transpose of matrix."""

        return MatrixN(self.vals.swapaxes(-2,-1), self.mask)

    ####################################################
    # Overrides of multiplication operators
    ####################################################

    def __mul__(self, arg):

        if isinstance(arg, Array):

            # MatrixN * any matrix is standard matrix multiply
            if arg.rank == 2:
                result = self.multiply_matrix(arg)

            # MatrixN * any vector is matrix multiply
            elif arg.rank == 1:
                result = self.multiply_vector(arg)

            # Anything else is treated as Scalar multiply
            else:
                return Array.__mul__(self, arg)

            # Multiply subfields if necessary
            result.mul_subfields(self, arg)

            return result

        # Otherwise, perform a standard multiply
        return Array.__mul__(self, arg)

    def __rmul__(self, arg):

        # VectorN * MatrixN is matrix post-multiply
        if isinstance(arg, Array) and arg.rank == 1:
            return VectorN(arg).as_row() * arg

        # Otherwise, only scalar multiply is allowed
        return Array.__mul__(self, Scalar.as_scalar(arg))

    def __imul__(self, arg):

        # MatrixN *= matrix is in-place matrix multiply
        if isinstance(arg, Array) and arg.rank == 2:
            if self.item != arg.item or self.item[0] != self.item[1]:
                raise ValueError("in-place multiply requires square matrices " +
                                 "of the same shape: " + str(self.item) +
                                 " *= " + str(arg.item))

            result = self * arg
            self.vals[...] = result[...]
            self.mask |= arg.mask

            self.imul_subfields(arg)

            return self

        # Otherwise, only scalar multiply is allowed
        return Array.__imul__(self, Scalar.as_scalar(arg))

    ####################################################
    # Overrides of division operators
    ####################################################

    def __div__(self, arg):
        return Scalar.as_scalar(arg).__rdiv__(self)

    def __rdiv__(self, arg):
        raise ValueError("MatrixN division is not supported")

    def __idiv__(self, arg):
        raise ValueError("MatrixN division is not supported")

    def __invert__(self):
        return self.inverse()

    ####################################################
    # Overrides of boolean operators
    ####################################################

    # A VectorN can be equal to either a row or column MatrixN
    def __eq__(self, arg):
        if isinstance(arg, (VectorN,Vector3)):
            if self.item[0] == 1: return self == arg.as_row()
            if self.item[1] == 1: return self == arg.as_column()

        return Array.__eq__(self, arg)

    def __ne__(self, arg):
        if isinstance(arg, (VectorN,Vector3)):
            if arg.item[0] == 1: return self != arg.as_row()
            if arg.item[1] == 1: return self != arg.as_column()

        return Array.__ne__(self, arg)

################################################################################
# Once defined, register with VectorN class
################################################################################

Array.MATRIXN_CLASS = MatrixN

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_MatrixN(unittest.TestCase):

    def runTest(self):

        a = VectorN((1,2))
        b = VectorN((0,1,-1))
        ab = a * b

        self.assertEqual(ab, MatrixN([(0.,1.,-1.),
                                      (0.,2.,-2.)]))

        self.assertEqual(ab * VectorN((3,2,1)), VectorN([1.,2.]))
        self.assertEqual(ab * VectorN([(3,2,1),
                                       (1,2,0)]), VectorN(([1.,2.],
                                                           [2.,4.])))

        v = VectorN([(3,2,1),(1,2,0)])
        self.assertEqual(v.shape, [2])
        self.assertEqual(v.item, [3])
        self.assertEqual(v*2, VectorN([(6,4,2),(2,4,0)]))
        self.assertEqual(v/2, VectorN([(1.5,1.,0.5),(0.5,1.,0.)]))
        self.assertEqual(2*v, 2.*v)

        m = MatrixN([(3,2,1),(1,2,0)])
        self.assertEqual(m.shape, [])
        self.assertEqual(m.item, [2,3])
        self.assertEqual(m*2, MatrixN([(6,4,2),(2,4,0)]))
        self.assertEqual(m/2, MatrixN([(1.5,1.,0.5),(0.5,1.,0.)]))
        self.assertEqual(2*m, 2.*m)

        self.assertEqual(a*m, VectorN([5,6,1]))

        i = MatrixN([(-1,0,0),(0,2,0),(0,0,0)])
        self.assertEqual(m*i, MatrixN([(-3,4,0),(-1,4,0)]))
        self.assertEqual(i*v, VectorN([(-3,4,0),(-1,4,0)]))

        j = MatrixN([(-1,0),(0,2),(1,1)])
        self.assertEqual(j*m, MatrixN([(-3,-2,-1),(2,4,0),(4,4,1)]))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
