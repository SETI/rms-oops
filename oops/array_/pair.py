################################################################################
# oops_/array_/pair.py: Pair subclass of class Array
#
# Modified 12/12/2011 (BSW) - removed redundant floor calls on astype('int')
#                           - added some comments
#
# Modified 1/2/11 (MRS) -- Uses a cleaner style of imports.
# Modified 1/12/11 (MRS) -- Added method cross_scalars()
# Modified 2/8/12 (MRS) -- Supports array masks; includes new unit tests.
# 3/2/12 MRS: Integrated with VectorN and MatrixN
################################################################################

import numpy as np
import numpy.ma as ma

from oops_.array.array_  import Array
from oops_.array.scalar  import Scalar
from oops_.array.vectorn import VectorN
from oops_.units import Units
import oops_.array.utils as utils

class Pair(Array):
    """An arbitrary Array of coordinate pairs or 2-vectors.
    """

    def __init__(self, arg, mask=False, units=None):

        return Array.__init__(self, arg, mask, units, 1, item=[2],
                                    floating=False, dimensionless=False)

    @staticmethod
    def as_pair(arg):
        if isinstance(arg, Pair): return arg

        # Collapse a 1x2 or 2x1 MatrixN down to a Pair
        if isinstance(arg, Array.MATRIXN_CLASS):
            return Pair.as_pair(VectorN.as_vectorn(arg))

        # If a single value is provided, duplicate it
        try:
            if np.shape(arg) == ():
                return Pair((arg,arg))
        except: pass

        return Pair(arg)

    @staticmethod
    def as_float(arg, copy=False):
        """Convert to float if necessary; copy=True to return a new copy."""

        if isinstance(arg, Pair):
            if isinstance(arg.vals, np.ndarray):
                if arg.vals.dtype == np.dtype("float"):
                    if copy:
                        return arg.copy()
                    else:
                        return arg

                return Pair(arg.vals.astype("float"), arg.mask, arg.units)

            else:
                if type(arg.vals) == type(0.):
                    if copy:
                        return arg.copy()
                    else:
                        return arg

                return Pair(float(arg.vals), arg.mask, arg.units)

        return Pair.as_float(Pair.as_pair(arg))

    @staticmethod
    def as_int(arg, copy=False):
        """Convert to int if necessary; copy=True to return a new copy."""

        if isinstance(arg, Pair):
            if isinstance(arg.vals, np.ndarray):
                if arg.vals.dtype == np.dtype("int"):
                    if copy:
                        return arg.copy()
                    else:
                        return arg

                return Pair(arg.vals.astype("int"), arg.mask, arg.units)

            else:
                if type(arg.vals) == type(0):
                    if copy:
                        return arg.copy()
                    else:
                        return arg

                return Pair(int(arg.vals), arg.mask, arg.units)

        return Pair.as_int(Pair.as_pair(arg))

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Pair): arg = Pair(arg)
        return arg.convert_units(None)

    def as_scalar(self, axis=0):
        """Overrides the defaul as_scalar method to include the axis argument.
        Returns one of the components of a Pair as a Scalar.

        Input:
            axis        0 for the x-axis; 1 for the y-axis.
        """

        return Scalar(self.vals[...,axis], self.mask, self.units)

    def as_scalars(self):
        """Returns the components of a Pair as a pair of Scalars.
        """

        return (Scalar(self.vals[...,0], self.mask, self.units),
                Scalar(self.vals[...,1], self.mask, self.units))

    @staticmethod
    def from_scalars(x,y):
        """Returns a new Pair constructed by combining the pairs of x- and y-
        components provided as scalars.
        """

        x = Scalar.as_scalar(x)
        y = Scalar.as_scalar(y).confirm_units(x.units)
        (x,y) = Array.broadcast_arrays(x,y)

        buffer = np.empty(x.shape + [2])
        buffer[...,0] = x.vals
        buffer[...,1] = y.vals

        return Pair(buffer, x.mask | y.mask, x.units)

    def as_column(self):
        """Converts the vector to an 2x1 column matrix."""

        return VectorN(self).as_column()

    def as_row(self):
        """Converts the vector to a 1x2 row matrix."""

        return VectorN(self).as_row()

    @staticmethod
    def cross_scalars(x,y):
        """Returns a new Pair constructed by combining every possible pairs of
        x- and y-components provided as scalars. The returned pair will have a
        shape defined by concatenating the shapes of the x and y arrays.
        """

        x = Scalar.as_scalar(x)
        y = Scalar.as_scalar(y).confirm_units(x.units)

        newshape = x.shape + y.shape
        if x.vals.dtype.kind != "f" and y.vals.dtype.kind != "f":
            dtype = "int"
        else:
            dtype = "float"

        buffer = np.empty(newshape + [2], dtype=dtype)

        x = x.reshape(x.shape + len(y.shape) * [1])
        buffer[...,0] = x.vals
        buffer[...,1] = y.vals

        return Pair(buffer, x.mask | y.mask)

    def swapxy(self):
        """Returns a pair object in which the first and second values are
            switched.
        """

        return Pair(self.vals[..., -1::-1], self.mask, self.units)

    def as_index(self):
        """Returns this object in a form suitable for indexing another object.
        """

        return list(np.rollaxis((self.vals // 1).astype("int"), -1, 0))

    def int(self):
        """Returns the integer (floor) component of each index."""

        return Pair((self.vals // 1.).astype("int"), self.mask)

    def frac(self):
        """Returns the fractional component of each index."""

        return Pair(self.vals % 1., self.mask)

    def float(self):
        """Returns the same Pair but containing floating-point values."""

        return Pair.as_float(self, copy=False)

    def dot(self, arg):
        """Returns the dot products of two Pairs as a Scalar.
        """

        arg = Pair.as_pair(arg)
        return Scalar(utils.dot(self.vals, arg.vals),
                      self.mask | arg.mask,
                      Units.mul_units(self.units, arg.units))

    def norm(self):
        """Returns the length of the Pair as a Scalar.
        """

        return Scalar(utils.norm(self.vals), self.mask, self.units)

    def __abs__(self): return self.norm()

    def unit(self):
        """Returns a the Pair converted to unit length as a new Pair.
        """

        return Pair(utils.unit(self.vals), self.mask)

    def cross(self, arg):
        """Returns the magnitude of the cross products of the Pairs as a new
        Scalar.
        """

        arg = Pair.as_pair(arg)
        return Scalar(utils.cross2d(self.vals, arg.vals),
                      self.mask | arg.mask,
                      Units.mul_units(self.units, arg.units))


    def sep(self, arg):
        """Returns returns angle between two Pairs as a Scalar.
        """

        arg = Pair.as_pair(arg)
        return Scalar(utils.sep(self.vals, arg.vals), self.mask | arg.mask)

    @staticmethod
    def meshgrid(arg1, arg2):
        """Returns a new Pair constructed by combining every possible set of
        components provided as a list of scalars. The returned Pair will have a
        shape defined by concatenating the shapes of the arguments.
        """

        tuple = Array.TUPLE_CLASS.meshgrid(arg1, arg2)
        return Pair.as_pair(tuple)

# Useful class constants
Pair.ZERO  = Pair([0.,0.])
Pair.XAXIS = Pair([1.,0.])
Pair.YAXIS = Pair([0.,1.])
Pair.ONES  = Pair([1.,1.])

################################################################################
# Once defined, register with Array class
################################################################################

Array.PAIR_CLASS = Pair

################################################################################
# UNIT TESTS
################################################################################

import unittest

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
        self.assertRaises(ValueError, pairs.__add__, Scalar(2))
        self.assertRaises(ValueError, pairs.__sub__, Scalar(2))

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

        # New tests 2/1/12 (MRS)

        test = Pair(np.arange(6).reshape(3,2))
        self.assertEqual(str(test), "Pair[[0 1]\n [2 3]\n [4 5]]")

        test.mask = np.array([False, False, True])
        self.assertEqual(str(test),   "Pair[[0 1]\n [2 3]\n [-- --], mask]")
        self.assertEqual(str(test*2), "Pair[[0 2]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test/2), "Pair[[0 0]\n [1 1]\n [-- --], mask]")
        self.assertEqual(str(test%2), "Pair[[0 1]\n [0 1]\n [-- --], mask]")

        self.assertEqual(str(test + (1,0)),
                         "Pair[[1 1]\n [3 3]\n [-- --], mask]")
        self.assertEqual(str(test - (0,1)),
                         "Pair[[0 0]\n [2 2]\n [-- --], mask]")
        self.assertEqual(str(test + test),
                         "Pair[[0 2]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test + np.arange(6).reshape(3,2)),
                         "Pair[[0 2]\n [4 6]\n [-- --], mask]")

        temp = Pair(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(test + temp),
                         "Pair[[-- --]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test - 2*temp),
                         "Pair[[-- --]\n [-2 -3]\n [-- --], mask]")
        self.assertEqual(str(test * temp),
                         "Pair[[-- --]\n [4 9]\n [-- --], mask]")
        self.assertEqual(str(test / temp),
                         "Pair[[-- --]\n [1 1]\n [-- --], mask]")
        self.assertEqual(str(test % temp),
                         "Pair[[-- --]\n [0 0]\n [-- --], mask]")
        self.assertEqual(str(test / [[2,1],[1,0],[7,0]]),
                         "Pair[[0 1]\n [-- --]\n [-- --], mask]")
        self.assertEqual(str(test % [[2,1],[1,0],[7,0]]),
                         "Pair[[0 0]\n [-- --]\n [-- --], mask]")

        temp = Pair(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(temp),      "Pair[[-- --]\n [2 3]\n [4 5], mask]")
        self.assertEqual(str(temp[0]),   "Pair[-- --, mask]")
        self.assertEqual(str(temp[1]),   "Pair[2 3]")
        self.assertEqual(str(temp[0:2]), "Pair[[-- --]\n [2 3], mask]")
        self.assertEqual(str(temp[0:1]), "Pair[[-- --], mask]")
        self.assertEqual(str(temp[1:2]), "Pair[[2 3]]")

        test = Pair(np.arange(6).reshape(3,2))
        self.assertEqual(test, Pair(np.arange(6).reshape(3,2)))

        mvals = test.mvals
        self.assertEqual(mvals.mask, ma.nomask)
        self.assertEqual(test, mvals)

        test.mask = np.array([False, False, True])
        mvals = test.mvals
        self.assertEqual(str(mvals), "[[0 1]\n [2 3]\n [-- --]]")
        self.assertEqual(test.mask.shape, (3,))
        self.assertEqual(mvals.mask.shape, (3,2))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
