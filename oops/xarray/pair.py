################################################################################
# Pair
#
# Modified 12/12/2011 (BSW) - removed redundant floor calls on astype('int')
#                           - added some comments
#
# Modified 1/2/11 (MRS) -- Uses a cleaner style of imports.
# Modified 1/12/11 (MRS) -- Added method cross_scalars()
################################################################################

import numpy as np
import unittest

from oops.broadcastable.Array  import Array
from oops.broadcastable.Scalar import Scalar

from oops import utils

class Pair(Array):
    """An arbitrary Array of coordinate pairs or 2-vectors.
    """

    OOPS_CLASS = "Pair"

    def __init__(self, arg):

        if isinstance(arg, Pair): return self.__init__(arg.vals)

        self.vals = np.asarray(arg)
        ashape = list(self.vals.shape)

        self.rank  = 1
        self.item  = ashape[-1:]
        self.shape = ashape[:-1]

        if self.item != [2]:
            raise ValueError("shape of a Pair array must be [...,2]")

        return

    @staticmethod
    def as_pair(arg, duplicate=False):
        if isinstance(arg, Pair): return arg

        if duplicate and np.shape(arg) == (): return Pair((arg,arg))
        return Pair(arg)

    @staticmethod
    def as_float_pair(arg, duplicate=False):
        if isinstance(arg, Pair) and arg.vals.dtype == np.dtype("float"):
            return arg

        return Pair.as_pair(arg, duplicate) * 1.

    def as_scalar(self, axis):
        """Returns one of the components of a Pair as a Scalar.

        Input:
            axis        0 for the x-axis; 1 for the y-axis.
        """

        return Scalar(self.vals[...,axis])

    def as_scalars(self):
        """Returns the components of a Pair as a pair of Scalars.
        """

        return (Scalar(self.vals[...,0]), Scalar(self.vals[...,1]))

    @staticmethod
    def from_scalars(x,y):
        """Returns a new Pair constructed by combining the pairs of x- and y-
        components provided as scalars.
        """

        (x,y) = Array.broadcast_arrays((Scalar.as_scalar(x),
                                        Scalar.as_scalar(y)))
        return Pair(np.vstack((x.vals,y.vals)).swapaxes(0,-1))

    @staticmethod
    def cross_scalars(x,y):
        """Returns a new Pair constructed by combining every possible pairs of
        x- and y-components provided as scalars. The returned pair will have a
        shape defined by concatenating the shapes of the x and y arrays.
        """

        x = Scalar.as_scalar(x)
        y = Scalar.as_scalar(y)

        newshape = x.shape + y.shape
        if x.vals.dtype.kind != "f" and y.vals.dtype.kind != "f":
            dtype = "int"
        else:
            dtype = "float"

        buffer = np.empty(newshape + [2], dtype=dtype)

        x = x.reshape(x.shape + len(y.shape) * [1])
        buffer[...,0] = x.vals
        buffer[...,1] = y.vals

        return Pair(buffer)

    def swapxy(self):
        """Returns a pair object in which the first and second values are
            switched.
        """

        return Pair(self.vals[..., -1::-1])

    def as_index(self):
        """Returns this object as a list of lists, which can be used to index a
        numpy ndarray, thereby returning an ndarray of the same shape as the
        Tuple object. Each value is rounded down to the nearest integer."""

        return list(np.rollaxis(self.vals.astype("int"),-1,0))

    def int(self):
        """Returns the integer (floor) component of each index."""

        return Pair(self.vals.astype("int"))

    def frac(self):
        """Returns the fractional component of each index."""

        return Pair(self.vals - np.floor(self.vals))

    def dot(self, arg):
        """Returns the dot products of two Pairs as a Scalar.
        """

        if isinstance(arg, Pair): arg = arg.vals
        return Scalar(utils.dot(self.vals, arg))

    def norm(self):
        """Returns the length of the Pair as a Scalar.
        """

        return Scalar(utils.norm(self.vals))

    def unit(self):
        """Returns a the Pair converted to unit length as a new Pair.
        """

        return Pair(utils.unit(self.vals))

    def cross(self, arg):
        """Returns the magnitude of the cross products of the Pairs as a new
        Scalar.
        """

        if isinstance(arg, Pair):
            arg = arg.vals
        else:
            if np.shape(arg)[-1] != 2:
                raise ValueError("shape of a Pair array must be [...,2]")

        return Scalar(utils.cross2d(self.vals,arg))

    def sep(self, arg):
        """Returns returns angle between two Pairs as a Scalar.
        """

        if isinstance(arg, Pair):
            arg = arg.vals
        else:
            if np.shape(arg)[-1] != 2:
                raise ValueError("shape of a Pair array must be [...,2]")

        return Scalar(utils.sep(self.vals,arg))

########################################
# UNIT TESTS
########################################

class Test_Pair(unittest.TestCase):

    def runTest(self):

        eps = 3.e-16
        lo = 1. - eps
        hi = 1. + eps

        # Basic comparisons and indexing
        pairs = Pair([[1,2],[3,4],[5,6]])
        self.assertEqual(Array.item(pairs),  [2])
        self.assertEqual(Array.shape(pairs), [3])
        self.assertEqual(Array.rank(pairs),   1)

        test = [[1,2],[3,4],[5,6]]
        self.assertEqual(pairs, test)

        test = Pair(test)
        self.assertEqual(pairs, test)

        self.assertTrue(pairs == test)
        self.assertTrue(not (pairs !=  test))
        self.assertTrue(~(pairs != test))

        self.assertEqual((pairs == test), True)
        self.assertEqual((pairs != test), False)
        self.assertEqual((pairs == test), (True,  True,  True))
        self.assertEqual((pairs != test), (False, False, False))
        self.assertEqual((pairs == test), Scalar(True))
        self.assertEqual((pairs != test), Scalar(False))
        self.assertEqual((pairs == test), Scalar((True,  True,  True)))
        self.assertEqual((pairs != test), Scalar((False, False, False)))

        self.assertEqual(pairs[0], (1,2))
        self.assertEqual(pairs[0], [1,2])
        self.assertEqual(pairs[0], Pair([1,2]))

        self.assertEqual(pairs[0:1], ((1,2)))
        self.assertEqual(pairs[0:1], [[1,2]])
        self.assertEqual(pairs[0:1], Pair([[1,2]]))

        self.assertEqual(pairs[0:2], ((1,2),(3,4)))
        self.assertEqual(pairs[0:2], [[1,2],[3,4]])
        self.assertEqual(pairs[0:2], Pair([[1,2],[3,4]]))

        # Unary operations
        self.assertEqual(+pairs, pairs)
        self.assertEqual(-pairs, Pair([[-1,-2],[-3,-4],(-5,-6)]))

        # Binary operations
        self.assertEqual(pairs + (2,2), [[3,4],[5,6],(7,8)])
        self.assertEqual(pairs + (2,2), Pair([[3,4],[5,6],(7,8)]))
        self.assertEqual(pairs - (2,2), [[-1,0],[1,2],[3,4]])
        self.assertEqual(pairs - (2,2), Pair([[-1,0],[1,2],[3,4]]))

        self.assertEqual(pairs * (2,2), [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * (2,2), Pair([[2,4],[6,8],[10,12]]))
        self.assertEqual(pairs * (1,2), [[1,4],[3,8],[5,12]])
        self.assertEqual(pairs * (1,2), Pair([[1,4],[3,8],[5,12]]))
        self.assertEqual(pairs * Pair((1,2)), [[1,4],[3,8],[5,12]])
        self.assertEqual(pairs * Pair((1,2)), Pair([[1,4],[3,8],[5,12]]))
        self.assertEqual(pairs * 2, [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * 2, [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * Scalar(2), [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * Scalar(2), [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * (1,2,3), [[1,2],[6,8],[15,18]])
        self.assertEqual(pairs * Scalar((1,2,3)), [[1,2],[6,8],[15,18]])

        self.assertEqual(pairs / (2,2), [[0,1],[1,2],[2,3]])
        self.assertEqual(pairs / (2,2), Pair([[0,1],[1,2],[2,3]]))
        self.assertEqual(pairs / (1,2), [[1,1],[3,2],[5,3]])
        self.assertEqual(pairs / (1,2), Pair([[1,1],[3,2],[5,3]]))
        self.assertEqual(pairs / Pair((1,2)), [[1,1],[3,2],[5,3]])
        self.assertEqual(pairs / Pair((1,2)), Pair([[1,1],[3,2],[5,3]]))
        self.assertEqual(pairs / 2, [[0,1],[1,2],[2,3]])
        self.assertEqual(pairs / 2, Pair([[0,1],[1,2],[2,3]]))
        self.assertEqual(pairs / Scalar(2), [[0,1],[1,2],[2,3]])
        self.assertEqual(pairs / Scalar(2), Pair([[0,1],[1,2],[2,3]]))
        self.assertEqual(pairs / (1,2,3), [[1,2],[1,2],[1,2]])
        self.assertEqual(pairs / Scalar((1,2,3)), [[1,2],[1,2],[1,2]])

        self.assertRaises(ValueError, pairs.__add__, 2)
        self.assertRaises(ValueError, pairs.__sub__, 2)
        self.assertRaises(TypeError, pairs.__add__, Scalar(2))
        self.assertRaises(TypeError, pairs.__sub__, Scalar(2))

        # In-place operations
        test = pairs.copy()
        test += (2,2)
        self.assertEqual(test, [[3,4],[5,6],(7,8)])
        test -= (2,2)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= (1,2)
        self.assertEqual(test, [[1,4],[3,8],[5,12]])
        test /= (1,2)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= (1,2,3)
        self.assertEqual(test, [[1,2],[6,8],[15,18]])
        test /= (1,2,3)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= 2
        self.assertEqual(test, [[2,4],[6,8],[10,12]])
        test /= 2
        self.assertEqual(test, [[1,2],[3,4],[5,6]])

        test += Pair((2,2))
        self.assertEqual(test, [[3,4],[5,6],(7,8)])
        test -= Pair((2,2))
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= Pair((1,2))
        self.assertEqual(test, [[1,4],[3,8],[5,12]])
        test /= Pair((1,2))
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= Scalar((1,2,3))
        self.assertEqual(test, [[1,2],[6,8],[15,18]])
        test /= Scalar((1,2,3))
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= Scalar(2)
        self.assertEqual(test, [[2,4],[6,8],[10,12]])
        test /= Scalar(2)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])

        # Other functions...

        # as_scalar()
        self.assertEqual(pairs.as_scalar(0),  Scalar((1,3,5)))
        self.assertEqual(pairs.as_scalar(1),  Scalar((2,4,6)))
        self.assertEqual(pairs.as_scalar(-1), Scalar((2,4,6)))
        self.assertEqual(pairs.as_scalar(-2), Scalar((1,3,5)))

        # as_scalars()
        self.assertEqual(pairs.as_scalars(), (Scalar((1,3,5)),
                                              Scalar((2,4,6))))

        # swapxy()
        self.assertEqual(pairs.swapxy(), Pair(((2,1),(4,3),(6,5))))

        # dot()
        self.assertEqual(pairs.dot((1,0)), pairs.as_scalar(0))
        self.assertEqual(pairs.dot((0,1)), pairs.as_scalar(1))
        self.assertEqual(pairs.dot((1,1)),
                         pairs.as_scalar(0) + pairs.as_scalar(1))

        # norm()
        self.assertEqual(pairs.norm(), np.sqrt((5.,25.,61.)))
        self.assertEqual(pairs.norm(), Scalar(np.sqrt((5.,25.,61.))))

        self.assertTrue(pairs.unit().norm() > lo)
        self.assertTrue(pairs.unit().norm() < hi)
        self.assertTrue(pairs.sep(pairs.unit()) > -eps)
        self.assertTrue(pairs.sep(pairs.unit()) <  eps)

        # cross()
        axes = Pair([(1,0),(0,1)])
        axes2 = axes.reshape((2,1))
        self.assertEqual(axes.cross(axes2), [[0,-1],[1,0]])

        # sep()
        self.assertTrue(axes.sep((1,1)) > np.pi/4. - eps)
        self.assertTrue(axes.sep((1,1)) < np.pi/4. + eps)

        angles = np.arange(0., np.pi, 0.01)
        vecs = Pair.from_scalars(np.cos(angles), np.sin(angles))
        self.assertTrue(Pair([2,0]).sep(vecs) > angles - 3*eps)
        self.assertTrue(Pair([2,0]).sep(vecs) < angles + 3*eps)

        vecs = Pair.from_scalars(np.cos(angles), -np.sin(angles))
        self.assertTrue(Pair([2,0]).sep(vecs) > angles - 3*eps)
        self.assertTrue(Pair([2,0]).sep(vecs) < angles + 3*eps)

        # cross_scalars()
        pair = Pair.cross_scalars(np.arange(10), np.arange(5))
        self.assertEqual(pair.shape, [10,5])
        self.assertTrue(np.all(pair.vals[9,:,0] == 9))
        self.assertTrue(np.all(pair.vals[:,4,1] == 4))

        pair = Pair.cross_scalars(np.arange(12).reshape(3,4), np.arange(5))
        self.assertEqual(pair.shape, [3,4,5])
        self.assertTrue(np.all(pair.vals[2,3,:,0] == 11))
        self.assertTrue(np.all(pair.vals[:,:,4,1] == 4))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
