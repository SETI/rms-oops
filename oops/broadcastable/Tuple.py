################################################################################
# Tuple
#
# Created 1/12/11 (MRS)
################################################################################

import numpy as np
import unittest

from oops.broadcastable.Array  import Array
from oops.broadcastable.Scalar import Scalar
from oops.broadcastable.Pair   import Pair

from oops import utils

################################################################################
# Tuple
################################################################################

class Tuple(Array):
    """An arbitrary Array of tuples, all of the same length."""

    OOPS_CLASS = "TUPLE"

    def __init__(self, arg):

        if isinstance(arg, Array):
            self.vals = arg.vals
        else:
            self.vals = np.asarray(arg)

        ashape = list(self.vals.shape)

        self.rank  = 1
        self.item  = ashape[-1:]
        self.shape = ashape[:-1]

        return

    @staticmethod
    def as_tuple(arg):
        if isinstance(arg, Tuple): return arg
        return Tuple(arg)

    def as_scalar(self, axis):
        """Returns a Scalar containing one selected item from each tuple."""

        return Scalar(self.vals[...,axis])

    def as_scalars(self):
        """Returns this object as a list of Scalars."""

        list = []
        for i in range(self.item[0]):
            list.append(Scalar(self.vals[...,i]))

        return list

    def as_pair(self, axis=0):
        """Returns a Pair containing two selected items from each Tuple,
        beginning with the selected axis."""

        return Pair(self.vals[...,axis:axis+2])

    @staticmethod
    def from_scalars(*args):
        """Returns a new Tuple constructed by combining the Scalars or arrays
        given as arguments.
        """

        return Tuple(np.rollaxis(np.array(args), 0, len(args)))

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

########################################
# UNIT TESTS
########################################

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

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
