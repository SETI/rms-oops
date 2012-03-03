################################################################################
# oops_/array_/tuple.py: Tuple subclass of class Array
#
# Created 1/12/12 (MRS)
# Modified 2/8/12 (MRS) -- Supports array masks; includes new unit tests.
################################################################################

import numpy as np
import numpy.ma as ma

from baseclass  import Array
from scalar     import Scalar
from pair       import Pair
from vector3    import Vector3

from oops_.units import Units

import utils

class Tuple(Array):
    """An arbitrary Array of tuples, all of the same length. Tuples and VectorN
    objects differ in the methods available and the way they perform certain
    arithmetic operations. Tuples are generally intended for indexing arrays,
    whereas VectorN objects are typically coupled with MatrixN operations."""

    def __init__(self, arg, mask=False, units=None):

        return Array.__init__(self, arg, mask, units, 1, item=None,
                                    float=False, dimensionless=True)

    @staticmethod
    def as_tuple(arg):
        if isinstance(arg, Tuple): return arg
        return Tuple(arg)

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Tuple): arg = Tuple(arg)
        return arg.convert_units(None)

    def as_scalar(self, axis):
        """Returns a Scalar containing one selected item from each tuple."""

        return Scalar(self.vals[...,axis], self.mask)

    def as_scalars(self):
        """Returns this object as a list of Scalars."""

        list = []
        for i in range(self.item[0]):
            list.append(Scalar(self.vals[...,i]), self.mask)

        return list

    def as_pair(self, axis=0):
        """Returns a Pair containing two selected items from each Tuple,
        beginning with the selected axis."""

        return Pair(self.vals[...,axis:axis+2], self.mask, self.units)

    def as_vector3(self, axis=0):
        """Returns a Vector3 containing three selected items from each Tuple,
        beginning with the selected axis."""

        return Vector3(self.vals[...,axis:axis+3], self.mask, self.units)

    @staticmethod
    def from_scalars(*args):
        """Returns a new Tuple constructed by combining the Scalars or arrays
        given as arguments.
        """

        scalars = []
        for arg in args:
            scalars.append(arg.as_scalar(arg))

        units = scalars[0].units
        mask  = scalars[0].mask

        arrays = [scalars[0].vals]
        for scalar in scalars[1:]:
            arrays.append(scalar.confirm_units(units).vals)
            mask = mask | scalar.mask

        return Tuple(np.rollaxis(np.array(arrays), 0, len(arrays)), mask, units)

    @staticmethod
    def cross_scalars(*args):
        """Returns a new Tuple constructed by combining every possible set of
        components provided as a list of scalars. The returned Tuple will have a
        shape defined by concatenating the shapes of all the arguments.
        """

        scalars = []
        newshape = []
        dtype = "int"
        for arg in args:
            scalar = Scalar.as_scalar(arg)
            scalars.append(scalar)
            newshape += scalar.shape
            if scalar.vals.dtype.kind == "f": dtype = "float"

        buffer = np.empty(newshape + [len(args)], dtype=dtype)

        newaxes = []
        count = 0
        for scalar in scalars[::-1]:
            newaxes.append(count)
            count += len(scalar.shape)

        newaxes.reverse()

        for i in range(len(scalars)):
            scalars[i] = scalars[i].reshape(scalars[i].shape + newaxes[i] * [1])

        reshaped = Array.broadcast_arrays(scalars)

        for i in range(len(reshaped)):
            buffer[...,i] = reshaped[i].vals

        return Tuple(buffer)

    @staticmethod
    def from_scalar_list(list):
        """Returns a new Tuple constructed by combining the Scalars or arrays
        given in a list.
        """

        return Tuple(np.rollaxis(np.array(list), 0, len(list)))

    def as_index(self):
        """Returns this object as a list of lists, which can be used to index a
        numpy ndarray, thereby returning an ndarray of the same shape as the
        Tuple object. Each value is rounded down to the nearest integer."""

        return list(np.rollaxis((self.vals // 1).astype("int"), -1, 0))

    def int(self):
        """Returns the integer (floor) component of each index."""

        return Tuple((self.vals // 1).astype("int"))

    def frac(self):
        """Returns the fractional component of each index."""

        return Tuple(self.vals % 1)

################################################################################
# Once defined, register with Array class
################################################################################

Array.TUPLE_CLASS = Tuple

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Tuple(unittest.TestCase):

    def runTest(self):

        foo = np.arange(24).reshape(3,4,2)

        test = Tuple(np.array([[[0,0,0], [0,0,1], [0,1,0]],
                               [[0,1,1], [0,2,0], [2,3,1]]]))
        self.assertEqual(test.shape, [2,3])
        self.assertEqual(test.item, [3])

        result = foo[test.as_index()]
        self.assertEqual(result.shape, (2,3))
        self.assertTrue(np.all(result == [[0, 1, 2],[3, 4, 23]]))

        self.assertEqual(test + (1,1,0), [[[1,1,0], [1,1,1], [1,2,0]],
                                          [[1,2,1], [1,3,0], [3,4,1]]])

        self.assertEqual((test + (0.5,0.5,0.5)).int(), test)

        self.assertTrue(np.all((test + (0.5,0.5,0.5)).frac().vals == 0.5))

        # cross_scalars()
        t = Tuple.cross_scalars(np.arange(5), np.arange(4), np.arange(3))
        self.assertEqual(t.shape, [5,4,3])
        self.assertTrue(np.all(t.vals[4,:,:,0] == 4))
        self.assertTrue(np.all(t.vals[:,3,:,1] == 3))
        self.assertTrue(np.all(t.vals[:,:,2,2] == 2))

        # cross_scalars()
        t = Tuple.cross_scalars(np.arange(5), np.arange(12).reshape(4,3),
                                np.arange(2))
        self.assertEqual(t.shape, [5,4,3,2])
        self.assertTrue(np.all(t.vals[4,:,:,:,0] ==  4))
        self.assertTrue(np.all(t.vals[:,3,2,:,1] == 11))
        self.assertTrue(np.all(t.vals[:,:,:,1,2] ==  1))

        # New tests 2/1/12 (MRS)

        test = Tuple(np.arange(6).reshape(3,2))
        self.assertEqual(str(test), "Tuple[[0 1]\n [2 3]\n [4 5]]")

        test.mask = np.array([False, False, True])
        self.assertEqual(str(test),   "Tuple[[0 1]\n [2 3]\n [-- --], mask]")
        self.assertEqual(str(test*2), "Tuple[[0 2]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test/2), "Tuple[[0 0]\n [1 1]\n [-- --], mask]")
        self.assertEqual(str(test%2), "Tuple[[0 1]\n [0 1]\n [-- --], mask]")

        self.assertEqual(str(test + (1,0)),
                         "Tuple[[1 1]\n [3 3]\n [-- --], mask]")
        self.assertEqual(str(test - (0,1)),
                         "Tuple[[0 0]\n [2 2]\n [-- --], mask]")
        self.assertEqual(str(test + test),
                         "Tuple[[0 2]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test + np.arange(6).reshape(3,2)),
                         "Tuple[[0 2]\n [4 6]\n [-- --], mask]")

        temp = Tuple(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(test + temp),
                         "Tuple[[-- --]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test - 2*temp),
                         "Tuple[[-- --]\n [-2 -3]\n [-- --], mask]")
        self.assertEqual(str(test * temp),
                         "Tuple[[-- --]\n [4 9]\n [-- --], mask]")
        self.assertEqual(str(test / temp),
                         "Tuple[[-- --]\n [1 1]\n [-- --], mask]")
        self.assertEqual(str(test % temp),
                         "Tuple[[-- --]\n [0 0]\n [-- --], mask]")
        self.assertEqual(str(test / [[2,1],[1,0],[7,0]]),
                         "Tuple[[0 1]\n [-- --]\n [-- --], mask]")
        self.assertEqual(str(test % [[2,1],[1,0],[7,0]]),
                         "Tuple[[0 0]\n [-- --]\n [-- --], mask]")

        temp = Tuple(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(temp),      "Tuple[[-- --]\n [2 3]\n [4 5], mask]")
        self.assertEqual(str(temp[0]),   "Tuple[-- --, mask]")
        self.assertEqual(str(temp[1]),   "Tuple[2 3]")
        self.assertEqual(str(temp[0:2]), "Tuple[[-- --]\n [2 3], mask]")
        self.assertEqual(str(temp[0:1]), "Tuple[[-- --], mask]")
        self.assertEqual(str(temp[1:2]), "Tuple[[2 3]]")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
