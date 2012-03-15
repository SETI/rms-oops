################################################################################
# oops_/array_/vector3.py: Vector3 subclass of class Array
#
# Modified 1/2/11 (MRS) -- Uses a cleaner style of imports.
# Modified 2/8/12 (MRS) -- Supports array masks; includes new unit tests.
# 3/2/12 MRS - Integrated with VectorN and MatrixN.
################################################################################

import numpy as np
import numpy.ma as ma

from oops_.array.array_  import Array
from oops_.array.scalar  import Scalar
from oops_.array.pair    import Pair
from oops_.array.vectorn import VectorN
from oops_.units import Units
import oops_.array.utils as utils

class Vector3(Array):
    """An arbitrary Array of 3-vectors."""

    def __init__(self, arg, mask=False, units=None):

        Array.__init__(self, arg, mask, units, 1, item=[3],
                                  float=True, dimensionless=False)

        self.x = self.vals[...,0]
        self.y = self.vals[...,1]
        self.z = self.vals[...,2]

        return

    @staticmethod
    def as_vector3(arg):
        if isinstance(arg, Vector3): return arg

        # Collapse a 1x3 or 3x1 MatrixN down to a Vector3
        if isinstance(arg, Array.MATRIXN_CLASS):
            return Vector3(VectorN(arg).as_vectorn())

        return Vector3(arg)

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Vector3): arg = Vector3(arg)
        return arg.convert_units(None)

    def as_scalar(self, axis):
        """Returns one of the components of a Vector3 as a Scalar.

        Input:
            axis        axis index.
        """

        return Scalar(self.vals[...,axis], self.mask, self.units)

    def as_scalars(self):
        """Returns the components of a Vector3 as a triplet of Scalars.
        """

        return (Scalar(self.vals[...,0], self.mask, self.units),
                Scalar(self.vals[...,1], self.mask, self.units),
                Scalar(self.vals[...,2], self.mask, self.units))

    @staticmethod
    def from_scalars(x,y,z):
        """Returns a new Vector3 constructed by combining the given x, y and z
        components provided as scalars.
        """

        x = Scalar.as_scalar(x)
        y = Scalar.as_scalar(y).confirm_units(x.units)
        z = Scalar.as_scalar(z).confirm_units(x.units)
        (xvals, yvals, zvals) = np.broadcast_arrays(x.vals, y.vals, z.vals)

        vals = np.empty(xvals.shape + (3,), dtype=xvals.dtype)
        vals[...,0] = xvals
        vals[...,1] = yvals
        vals[...,2] = zvals
        return Vector3(vals, x.mask | y.mask | z.mask, x.units)

    def dot(self, arg):
        """Returns the dot products of the vectors as a Scalar."""

        arg = Vector3.as_vector3(arg)
        return Scalar(utils.dot(self.vals, arg.vals), self.mask | arg.mask,
                                Units.mul_units(self.units, arg.units))

    def norm(self):
        """Returns the length of the Vector3 as a Scalar."""

        return Scalar(utils.norm(self.vals), self.mask, self.units)

    def __abs__(self): return self.norm()

    def unit(self):
        """Returns a the vector converted to unit length as a Vector3."""

        return Vector3(utils.unit(self.vals), self.mask)

    def cross(self, arg):
        """Returns the cross products of the vectors as a Vector3."""

        arg = Vector3.as_vector3(arg)
        return Vector3(utils.cross3d(self.vals, arg.vals),
                       self.mask | arg.mask,
                       Units.mul_units(self.units, arg.units))

    def ucross(self, arg):
        """Returns the unit vector in the direction of the cross products of the
        vectors as a Vector3."""

        arg = Vector3.as_vector3(arg)
        return Vector3(utils.ucross3d(self.vals, arg.vals),
                       self.mask | arg.mask)

    def perp(self, arg):
        """Returns the component of a Vector3 perpendicular to another Vector3.
        """

        arg = Vector3.as_vector3(arg)
        return Vector3(utils.perp(self.vals, arg.vals), self.mask | arg.mask,
                                                        self.units)

    def proj(self, arg):
        """Returns the component of a Vector3 projected into another Vector3."""

        arg = Vector3.as_vector3(arg)
        return Vector3(utils.proj(self.vals, arg.vals), self.mask | arg.mask,
                                                        self.units)

    def sep(self, arg, reversed=False):
        """Returns returns angle between two Vector3 objects as a Scalar."""

        arg = Vector3.as_vector3(arg)
        if reversed: arg.vals = -arg.vals

        return Scalar(utils.sep(self.vals, arg.vals), self.mask | arg.mask)

    def cross_product_as_matrix(self):
        """Returns a MatrixN object for which matrix multiply produces the same
        result as taking a cross product with this vector."""

        vals = np.zeros(self.shape + [3,3])
        vals[...,0,1] = -self.vals[...,2]
        vals[...,0,2] =  self.vals[...,1]
        vals[...,1,2] = -self.vals[...,0]
        vals[...,1,0] =  self.vals[...,2]
        vals[...,2,0] = -self.vals[...,1]
        vals[...,2,1] =  self.vals[...,0]

        return Array.MATRIXN_CLASS(vals, self.mask)

    def as_column(self):
        """Converts the vector to an 3x1 column matrix."""

        return VectorN(self).as_column()

    def as_row(self):
        """Converts the vector to a 1x3 row matrix."""

        return VectorN(self).as_row()

################################################################################
# Once defined, register with Array class
################################################################################

Array.VECTOR3_CLASS = Vector3

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Vector3(unittest.TestCase):

    def runTest(self):

        eps = 3.e-16
        lo = 1. - eps
        hi = 1. + eps

        # Basic comparisons and indexing
        vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])
        self.assertEqual(Array.item(vecs),  [3])
        self.assertEqual(Array.shape(vecs), [3])
        self.assertEqual(Array.rank(vecs),   1)

        test = [[1,2,3],[3,4,5],[5,6,7]]
        self.assertEqual(vecs, test)

        test = Vector3(test)
        self.assertEqual(vecs, test)

        self.assertTrue(vecs == test)
        self.assertTrue(not (vecs != test))

        self.assertEqual((vecs == test), True)
        self.assertEqual((vecs != test), False)
        self.assertEqual((vecs == test), (True,  True,  True))
        self.assertEqual((vecs != test), (False, False, False))
        self.assertEqual((vecs == test), Scalar(True))
        self.assertEqual((vecs != test), Scalar(False))
        self.assertEqual((vecs == test), Scalar((True,  True,  True)))
        self.assertEqual((vecs != test), Scalar((False, False, False)))

        self.assertEqual((vecs == [1,2,3]), Scalar((True, False, False)))

        self.assertEqual(vecs[0], (1,2,3))
        self.assertEqual(vecs[0], [1,2,3])
        self.assertEqual(vecs[0], Vector3([1,2,3]))

        self.assertEqual(vecs[0:1], ((1,2,3)))
        self.assertEqual(vecs[0:1], [[1,2,3]])
        self.assertEqual(vecs[0:1], Vector3([[1,2,3]]))

        self.assertEqual(vecs[0:2], ((1,2,3),(3,4,5)))
        self.assertEqual(vecs[0:2], [[1,2,3],[3,4,5]])
        self.assertEqual(vecs[0:2], Vector3([[1,2,3],[3,4,5]]))

        # Unary operations
        self.assertEqual(+vecs, vecs)
        self.assertEqual(-vecs, Vector3([[-1,-2,-3],[-3,-4,-5],(-5,-6,-7)]))

        # Binary operations
        self.assertEqual(vecs + (0,1,2), [[1,3,5],[3,5,7],(5,7,9)])
        self.assertEqual(vecs + (0,1,2), Vector3([[1,3,5],[3,5,7],(5,7,9)]))
        self.assertEqual(vecs - (0,1,2), [[1,1,1],[3,3,3],[5,5,5]])
        self.assertEqual(vecs - (0,1,2), Vector3([[1,1,1],[3,3,3],[5,5,5]]))

        self.assertEqual(vecs * (1,2,3), [[1,4,9],[3,8,15],[5,12,21]])
        self.assertEqual(vecs * (1,2,3), Vector3([[1,4,9],[3,8,15],[5,12,21]]))
        self.assertEqual(vecs * Vector3((1,2,3)), [[1,4,9],[3,8,15],[5,12,21]])
        self.assertEqual(vecs * Vector3((1,2,3)), Vector3([[1,4,9],[3,8,15],
                                                           [5,12,21]]))
        self.assertEqual(vecs * 2, [[2,4,6],[6,8,10],[10,12,14]])
        self.assertEqual(vecs * 2, Vector3([[2,4,6],[6,8,10],[10,12,14]]))
        self.assertEqual(vecs * Scalar(2), [[2,4,6],[6,8,10],[10,12,14]])
        self.assertEqual(vecs * Scalar(2), Vector3([[2,4,6],[6,8,10],
                                                                 [10,12,14]]))

        self.assertEqual(vecs / (1,1,2), [[1,2,1.5],[3,4,2.5],[5,6,3.5]])
        self.assertEqual(vecs / Vector3((1,1,2)), [[1,2,1.5],[3,4,2.5],
                                                             [5,6,3.5]])

        self.assertEqual(vecs / 2, [[0.5,1,1.5],[1.5,2,2.5],[2.5,3,3.5]])
        self.assertEqual(vecs / Scalar(2), [[0.5,1,1.5],[1.5,2,2.5],
                                                             [2.5,3,3.5]])

        self.assertRaises(ValueError, vecs.__add__, 1)
        self.assertRaises(ValueError, vecs.__add__, Scalar(1))
        self.assertRaises(ValueError, vecs.__add__, (1,2))
        self.assertRaises(ValueError, vecs.__add__, Pair((1,2)))

        self.assertRaises(ValueError, vecs.__sub__, 1)
        self.assertRaises(ValueError, vecs.__sub__, Scalar(1))
        self.assertRaises(ValueError, vecs.__sub__, (1,2))
        self.assertRaises(ValueError, vecs.__sub__, Pair((1,2)))

        self.assertRaises(ValueError, vecs.__mul__, (1,2))
        self.assertRaises(ValueError, vecs.__mul__, Pair((1,2)))

        self.assertRaises(ValueError, vecs.__div__, (1,2))
        self.assertRaises(ValueError, vecs.__div__, Pair((1,2)))

        # In-place operations
        test = vecs.copy()
        test += (1,2,3)
        self.assertEqual(test, [[2,4,6],[4,6,8],(6,8,10)])
        test -= (1,2,3)
        self.assertEqual(test, vecs)
        test *= (1,2,3)
        self.assertEqual(test, [[1,4,9],[3,8,15],[5,12,21]])
        test /= (1,2,3)
        self.assertEqual(test, vecs)
        test *= 2
        self.assertEqual(test, [[2,4,6],[6,8,10],[10,12,14]])
        test /= 2
        self.assertEqual(test, vecs)
        test *= Scalar(2)
        self.assertEqual(test, [[2,4,6],[6,8,10],[10,12,14]])
        test /= Scalar(2)
        self.assertEqual(test, vecs)

        test *= Scalar((1,2,3))
        self.assertEqual(test, [[1,2,3],[6,8,10],[15,18,21]])
        test /= Scalar((1,2,3))
        self.assertEqual(test, vecs)

        self.assertRaises(ValueError, test.__iadd__, Scalar(1))
        self.assertRaises(ValueError, test.__iadd__, 1)
        self.assertRaises(ValueError, test.__iadd__, (1,2))

        self.assertRaises(ValueError, test.__isub__, Scalar(1))
        self.assertRaises(ValueError, test.__isub__, 1)
        self.assertRaises(ValueError, test.__isub__, (1,2,3,4))

        self.assertRaises(ValueError, test.__imul__, Pair((1,2)))
        self.assertRaises(ValueError, test.__imul__, (1,2,3,4))

        self.assertRaises(ValueError, test.__idiv__, Pair((1,2)))
        self.assertRaises(ValueError, test.__idiv__, (1,2,3,4))

        # Other functions...

        # as_scalar()
        self.assertEqual(vecs.as_scalar(0),  Scalar((1,3,5)))
        self.assertEqual(vecs.as_scalar(1),  Scalar((2,4,6)))
        self.assertEqual(vecs.as_scalar(2),  Scalar((3,5,7)))
        self.assertEqual(vecs.as_scalar(-1), Scalar((3,5,7)))
        self.assertEqual(vecs.as_scalar(-2), Scalar((2,4,6)))
        self.assertEqual(vecs.as_scalar(-3), Scalar((1,3,5)))

        # as_scalars()
        self.assertEqual(vecs.as_scalars(), (Scalar((1,3,5)),
                                             Scalar((2,4,6)),
                                             Scalar((3,5,7))))

        # dot()
        self.assertEqual(vecs.dot((1,0,0)), vecs.as_scalar(0))
        self.assertEqual(vecs.dot((0,1,0)), vecs.as_scalar(1))
        self.assertEqual(vecs.dot((0,0,1)), vecs.as_scalar(2))
        self.assertEqual(vecs.dot((1,1,0)),
                         vecs.as_scalar(0) + vecs.as_scalar(1))

        # norm()
        v = Vector3([[[1,2,3],[2,3,4]],[[0,1,2],[3,4,5]]])
        self.assertEqual(v.norm(), np.sqrt([[14,29],[5,50]]))

        # cross(), ucross()
        a = Vector3([[[1,0,0]],[[0,2,0]],[[0,0,3]]])
        b = Vector3([ [0,3,3] , [2,0,2] , [1,1,0] ])
        axb = a.cross(b)

        self.assertEqual(a.shape,   [3,1])
        self.assertEqual(b.shape,     [3])
        self.assertEqual(axb.shape, [3,3])

        self.assertEqual(axb[0,0], ( 0,-3, 3))
        self.assertEqual(axb[0,1], ( 0,-2, 0))
        self.assertEqual(axb[0,2], ( 0, 0, 1))
        self.assertEqual(axb[1,0], ( 6, 0, 0))
        self.assertEqual(axb[1,1], ( 4, 0,-4))
        self.assertEqual(axb[1,2], ( 0, 0,-2))
        self.assertEqual(axb[2,0], (-9, 0, 0))
        self.assertEqual(axb[2,1], ( 0, 6, 0))
        self.assertEqual(axb[2,2], (-3, 3, 0))

        axb = a.ucross(b)
        self.assertEqual(axb[0,0], Vector3(( 0,-3, 3)).unit())
        self.assertEqual(axb[0,1], Vector3(( 0,-2, 0)).unit())
        self.assertEqual(axb[0,2], Vector3(( 0, 0, 1)).unit())
        self.assertEqual(axb[1,0], Vector3(( 6, 0, 0)).unit())
        self.assertEqual(axb[1,1], Vector3(( 4, 0,-4)).unit())
        self.assertEqual(axb[1,2], Vector3(( 0, 0,-2)).unit())
        self.assertEqual(axb[2,0], Vector3((-9, 0, 0)).unit())
        self.assertEqual(axb[2,1], Vector3(( 0, 6, 0)).unit())
        self.assertEqual(axb[2,2], Vector3((-3, 3, 0)).unit())

        # perp, proj, sep
        a = Vector3(np.random.rand(2,1,4,1,3))
        b = Vector3(np.random.rand(  3,4,2,3))

        aperp = a.perp(b)
        aproj = a.proj(b)

        self.assertEqual(aperp.shape, [2,3,4,2])
        self.assertEqual(aproj.shape, [2,3,4,2])

        eps = 3.e-14
        self.assertTrue(aperp.sep(b) > np.pi/2 - eps)
        self.assertTrue(aperp.sep(b) < np.pi/2 + eps)
        self.assertTrue(aproj.sep(b) % np.pi > -eps)
        self.assertTrue(aproj.sep(b) % np.pi <  eps)
        self.assertTrue(np.all((a - aperp - aproj).vals > -eps))
        self.assertTrue(np.all((a - aperp - aproj).vals <  eps))

        # Note: the sep(reverse=True) option is not tested here

        # New tests 2/1/12 (MRS)

        test = Vector3(np.arange(6).reshape(2,3))
        self.assertEqual(str(test), "Vector3[[ 0.  1.  2.]\n [ 3.  4.  5.]]")

        test.mask = np.array([True, False])
        self.assertEqual(str(test),
                "Vector3[[-- -- --]\n [3.0 4.0 5.0], mask]")
        self.assertEqual(str(test*2),
                "Vector3[[-- -- --]\n [6.0 8.0 10.0], mask]")
        self.assertEqual(str(test/2),
                "Vector3[[-- -- --]\n [1.5 2.0 2.5], mask]")

        self.assertEqual(str(test + (1,0,2)),
                "Vector3[[-- -- --]\n [4.0 4.0 7.0], mask]")
        self.assertEqual(str(test - (1,0,2)),
                "Vector3[[-- -- --]\n [2.0 4.0 3.0], mask]")
        self.assertEqual(str(test - 2*test),
                "Vector3[[-- -- --]\n [-3.0 -4.0 -5.0], mask]")
        self.assertEqual(str(test + np.arange(6).reshape(2,3)),
                "Vector3[[-- -- --]\n [6.0 8.0 10.0], mask]")

        self.assertEqual(str(test[0]),
                "Vector3[-- -- --, mask]")
        self.assertEqual(str(test[1]),
                "Vector3[ 3.  4.  5.]")
        self.assertEqual(str(test[0:2]),
                "Vector3[[-- -- --]\n [3.0 4.0 5.0], mask]")
        self.assertEqual(str(test[0:1]),
                "Vector3[[-- -- --], mask]")
        self.assertEqual(str(test[1:2]),
                "Vector3[[ 3.  4.  5.]]")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
