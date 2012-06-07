################################################################################
# oops_/array_/vectorn.py: VectorN subclass of class Array
#
# Created 3/2/12 (MRS)
################################################################################

import numpy as np
import numpy.ma as ma

from oops_.array.array_  import Array
from oops_.array.scalar  import Scalar
from oops_.units import Units
import oops_.array.utils as utils

class VectorN(Array):
    """An arbitrary Array of 1-D vectors, all of the same length. Tuples and
    VectorN objects differ in the methods available and the way they perform
    certain arithmetic operations. Tuples are generally intended for indexing
    arrays, whereas VectorN objects are typically coupled with MatrixN
    operations."""

    def __init__(self, arg, mask=False, units=None):

        return Array.__init__(self, arg, mask, units, 1, item=None,
                                    float=True, dimensionless=False)

    @staticmethod
    def as_vectorn(arg):
        if isinstance(arg, VectorN): return arg

        # Collapse a 1xN or Nx1 MatrixN down to a VectorN
        if isinstance(arg, Array.MATRIXN_CLASS):

            if arg.item[0] == 1:
                return VectorN(arg.vals.reshape(arg.shape + [arg.item[1]]),
                               arg.mask, arg.units)

            if arg.item[1] == 1:
                return VectorN(arg.vals.reshape(arg.shape + [arg.item[0]]),
                               arg.mask, arg.units)

        return VectorN(arg)

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, VectorN): arg = VectorN(arg)
        return arg.convert_units(None)

    @staticmethod
    def from_scalars(*args):
        """Returns a new VectorN constructed by combining the Scalars or arrays
        given as arguments.
        """

        return VectorN.as_vectorn(Tuple.from_scalars(*args))

    @staticmethod
    def from_scalar_list(list):
        """Returns a new Tuple constructed by combining the Scalars or arrays
        given in a list.
        """

        return VectorN.as_vectorn(Tuple.from_scalar_list(list))

    def as_column(self):
        """Converts the vector to an Nx1 column matrix."""

        return VectorN.MATRIXN_CLASS(self.vals[..., np.newaxis],
                                     self.mask, self.units)

    def as_row(self):
        """Converts the vector to a 1xN row matrix."""

        return VectorN.MATRIXN_CLASS(self.vals[..., np.newaxis, :],
                                     self.mask, self.units)

    def dot(self, arg):
        """Returns the dot products of the vectors as a Scalar."""

        arg = VectorN.as_vectorn(arg)
        return Scalar(np.sum(self.vals * arg.vals, axis=-1),
                             self.mask | arg.mask,
                             Units.mul_units(self.units, arg.units))

    def norm(self):
        """Returns the length of the VectorN as a Scalar."""

        return Scalar(np.sqrt(np.sum(self.vals**2, axis=-1)),
                      self.mask, self.units)

    def __abs__(self): return self.norm()

    def unit(self):
        """Returns a the vector converted to unit length as a VectorN."""

        return VectorN(self.vals /
                       np.sqrt(np.sum(self.vals**2, axis=-1))[..., np.newaxis],
                       self.mask)

    def cross(self, arg):
        """Returns the cross products of the vectors as a Vector3."""

        arg = VectorN.as_vectorn(arg)
        assert self.item == arg.item
        assert self.item[0] in (2,3)

        if self.item == [3]:
            return VectorN(utils.cross3d(self.vals, arg.vals),
                           self.mask | arg.mask,
                           Units.mul_units(self.units, arg.units))
        else:
            return VectorN(utils.cross2d(self.vals, arg.vals),
                           self.mask | arg.mask,
                           Units.mul_units(self.units, arg.units))

    def ucross(self, arg):
        """Returns the unit vector in the direction of the cross products of the
        vectors as a Vector3."""

        return self.cross(arg).unit()

    def perp(self, arg):
        """Returns the component of a Vector3 perpendicular to another Vector3.
        """

        arg = VectorN.as_vectorn(arg)
        unitvec = self.unit()
        return arg - unitvec * (unitvec.dot(arg))

    def proj(self, arg):
        """Returns the component of a Vector3 projected into another Vector3."""

        unitvec = self.unit()
        return unitvec * (unitvec.dot(arg))

    def cross_product_as_matrix(self):
        """Returns the MatrixN one would use multiply a vector of length 3 to
        yield the cross product of self with that vector. Self must have length
        3.
        """
        assert self.item == [3]

        vals = np.zeros(self.shape + [3,3])
        vals[...,0,1] = -self.vals[...,2]
        vals[...,0,2] =  self.vals[...,1]
        vals[...,1,2] = -self.vals[...,0]
        vals[...,1,0] =  self.vals[...,2]
        vals[...,2,0] = -self.vals[...,1]
        vals[...,2,1] =  self.vals[...,0]

        return Array.MATRIXN_CLASS(vals, self.mask)

    ####################################################
    # Overrides of arithmetic operators
    ####################################################

    def __mul__(self, arg):

        if isinstance(arg, Array):

            # Assume vector * matrix is pre-multiply
            if arg.rank == 2:
                return VectorN.as_vectorn(self.as_row() * arg)

            # Assume vector * vector is outer multiply
            if arg.rank == 1:
                return self.as_column() * VectorN.as_vectorn(arg).as_row()

        # Anything else is default multiply
        return Array.__mul__(self, arg)

    def __rmul__(self, arg):

        return self.__mul__(arg)

    # A VectorN can be equal to either a row or column MatrixN
    def __eq__(self, arg):
        if isinstance(arg, Array.MATRIXN_CLASS):
            if arg.item[0] == 1: return self.as_row() == arg
            if arg.item[1] == 1: return self.as_column() == arg

        return Array.__eq__(self, arg)

    def __ne__(self, arg):
        if isinstance(arg, Array.MATRIXN_CLASS):
            if arg.item[0] == 1: return self.as_row() != arg
            if arg.item[1] == 1: return self.as_column() != arg

        return Array.__ne__(self, arg)

# Useful class constants
VectorN.ZERO3 = VectorN([0,0,0])
VectorN.XAXIS = VectorN([1,0,0])
VectorN.YAXIS = VectorN([0,1,0])
VectorN.ZAXIS = VectorN([0,0,1])

################################################################################
# Once defined, register with Array class
################################################################################

Array.VECTORN_CLASS = VectorN

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_VectorN(unittest.TestCase):

    def runTest(self):

        omega = VectorN(np.random.rand(30,3))
        test = VectorN(np.random.rand(20,30,3))
        self.assertEqual(omega.cross(test),
                         omega.cross_product_as_matrix() * test)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
