################################################################################
# Old Pair tests, updated by MRS 2/18/14
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Pair, Scalar, Vector, Boolean, Units

class Test_Pair_misc(unittest.TestCase):

    def runTest(self):

        # Basic comparisons and indexing
        pairs = Pair([[1,2],[3,4],[5,6]])
        self.assertEqual(pairs.numer, (2,))
        self.assertEqual(pairs.shape, (3,))
        self.assertEqual(pairs.rank,     1)

        test = [[1,2],[3,4],[5,6]]
        self.assertEqual(pairs, test)

        test = Pair(test)
        self.assertEqual(pairs, test)

        self.assertTrue(pairs == test)
        self.assertTrue(not (pairs !=  test))
        self.assertTrue((~(pairs != test)).all())

        self.assertEqual((pairs == test).all(), True)
        self.assertEqual((pairs != test), False)
        self.assertEqual((pairs == test), (True,  True,  True))
        self.assertEqual((pairs != test), (False, False, False))
        self.assertEqual((pairs == test).all(), Scalar(True))
        self.assertEqual((pairs != test).all(), Scalar(False))
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
        pairs = Pair([[1,2],[3,4],[5,6]])
        self.assertEqual(pairs + (2,2), [[3,4],[5,6],(7,8)])
        self.assertEqual(pairs + (2,2), Pair([[3,4],[5,6],(7,8)]))
        self.assertEqual(pairs - (2,2), [[-1,0],[1,2],[3,4]])
        self.assertEqual(pairs - (2,2), Pair([[-1,0],[1,2],[3,4]]))

        self.assertEqual(pairs.element_mul((2,2)), [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs.element_mul((2,2)), Pair([[2,4],[6,8],[10,12]]))
        self.assertEqual(pairs.element_mul((1,2)), [[1,4],[3,8],[5,12]])
        self.assertEqual(pairs.element_mul((1,2)), Pair([[1,4],[3,8],[5,12]]))
        self.assertEqual(pairs.element_mul(Pair((1,2))), [[1,4],[3,8],[5,12]])
        self.assertEqual(pairs.element_mul(Pair((1,2))), Pair([[1,4],[3,8],[5,12]]))
        self.assertEqual(pairs * 2, [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * 2, [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * Scalar(2), [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * Scalar(2), [[2,4],[6,8],[10,12]])
        self.assertEqual(pairs * (1,2,3), [[1,2],[6,8],[15,18]])
        self.assertEqual(pairs * Scalar((1,2,3)), [[1,2],[6,8],[15,18]])

        self.assertEqual(pairs.element_div((2,2)), [[0.5,1],[1.5,2],[2.5,3]])
        self.assertEqual(pairs.element_div((2,2)), Pair([[0.5,1],[1.5,2],[2.5,3]]))
        self.assertEqual(pairs.element_div((1,2)), [[1,1],[3,2],[5,3]])
        self.assertEqual(pairs.element_div((1,2)), Pair([[1,1],[3,2],[5,3]]))
        self.assertEqual(pairs.element_div(Pair((1,2))), [[1,1],[3,2],[5,3]])
        self.assertEqual(pairs.element_div(Pair((1,2))), Pair([[1,1],[3,2],[5,3]]))
        self.assertEqual(pairs / 2, [[0.5,1],[1.5,2],[2.5,3]])
        self.assertEqual(pairs / 2, Pair([[0.5,1],[1.5,2],[2.5,3]]))
        self.assertEqual(pairs / Scalar(2), [[0.5,1],[1.5,2],[2.5,3]])
        self.assertEqual(pairs / Scalar(2), Pair([[0.5,1],[1.5,2],[2.5,3]]))
        self.assertEqual(pairs / (1,2,2), [[1,2],[1.5,2],[2.5,3]])
        self.assertEqual(pairs / Scalar((1,2,2)), [[1,2],[1.5,2],[2.5,3]])

        self.assertRaises(TypeError, pairs.__add__, 2)
        self.assertRaises(TypeError, pairs.__sub__, 2)
        self.assertRaises(TypeError, pairs.__add__, Scalar(2))
        self.assertRaises(TypeError, pairs.__sub__, Scalar(2))

        # In-place operations on ints
        test = pairs.copy()
        test += (2,2)
        self.assertEqual(test, [[3,4],[5,6],(7,8)])
        test -= (2,2)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= (1,2,3)
        self.assertEqual(test, [[1,2],[6,8],[15,18]])
        test //= (1,2,3)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= 2
        self.assertEqual(test, [[2,4],[6,8],[10,12]])
        test //= 2
        self.assertEqual(test, [[1,2],[3,4],[5,6]])

        test += Pair((2,2))
        self.assertEqual(test, [[3,4],[5,6],(7,8)])
        test -= Pair((2,2))
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= Scalar((1,2,3))
        self.assertEqual(test, [[1,2],[6,8],[15,18]])
        test //= Scalar((1,2,3))
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= Scalar(2)
        self.assertEqual(test, [[2,4],[6,8],[10,12]])
        test //= Scalar(2)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])

        # In-place operations on floats
        test = pairs.as_float()
        test += (2,2)
        self.assertEqual(test, [[3,4],[5,6],(7,8)])
        test -= (2,2)
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
        test *= Scalar((1,2,3))
        self.assertEqual(test, [[1,2],[6,8],[15,18]])
        test /= Scalar((1,2,3))
        self.assertEqual(test, [[1,2],[3,4],[5,6]])
        test *= Scalar(2)
        self.assertEqual(test, [[2,4],[6,8],[10,12]])
        test /= Scalar(2)
        self.assertEqual(test, [[1,2],[3,4],[5,6]])

        # Other functions...
        pairs = Pair([[1,2],[3,4],[5,6]])

        eps = 3.e-16
        lo = 1. - eps
        hi = 1. + eps

        # to_scalar()
        self.assertEqual(pairs.to_scalar(0),  Scalar((1,3,5)))
        self.assertEqual(pairs.to_scalar(1),  Scalar((2,4,6)))
        self.assertEqual(pairs.to_scalar(-1), Scalar((2,4,6)))
        self.assertEqual(pairs.to_scalar(-2), Scalar((1,3,5)))

        # to_scalars()
        self.assertEqual(pairs.to_scalars(), (Scalar((1,3,5)),
                                              Scalar((2,4,6))))

        # swapxy()
        self.assertEqual(pairs.swapxy(), Pair(((2,1),(4,3),(6,5))))

        # dot()
        self.assertEqual(pairs.dot((1,0)), pairs.to_scalar(0))
        self.assertEqual(pairs.dot((0,1)), pairs.to_scalar(1))
        self.assertEqual(pairs.dot((1,1)),
                         pairs.to_scalar(0) + pairs.to_scalar(1))

        # norm()
        self.assertEqual(pairs.norm(), np.sqrt((5.,25.,61.)))
        self.assertEqual(pairs.norm(), Scalar(np.sqrt((5.,25.,61.))))

        self.assertTrue((pairs.unit().norm() > lo).all())
        self.assertTrue((pairs.unit().norm() < hi).all())
        self.assertTrue((pairs.sep(pairs.unit()) > -eps).all())
        self.assertTrue((pairs.sep(pairs.unit()) <  eps).all())

        # cross()
        axes = Pair([(1,0),(0,1)])
        axes2 = axes.reshape((2,1))
        self.assertEqual(axes.cross(axes2), [[0,-1],[1,0]])

        # sep()
        self.assertTrue((axes.sep((1,1)) > np.pi/4. - eps).all())
        self.assertTrue((axes.sep((1,1)) < np.pi/4. + eps).all())

        angles = np.arange(0., np.pi, 0.01)
        vecs = Pair.from_scalars(np.cos(angles), np.sin(angles))
        self.assertTrue((Pair([2,0]).sep(vecs) > angles - 3*eps).all())
        self.assertTrue((Pair([2,0]).sep(vecs) < angles + 3*eps).all())

        vecs = Pair.from_scalars(np.cos(angles), -np.sin(angles))
        self.assertTrue((Pair([2,0]).sep(vecs) > angles - 3*eps).all())
        self.assertTrue((Pair([2,0]).sep(vecs) < angles + 3*eps).all())

        # cross_scalars()
#         pair = Pair.cross_scalars(np.arange(10), np.arange(5))
#         self.assertEqual(pair.shape, [10,5])
#         self.assertTrue(np.all(pair.vals[9,:,0] == 9))
#         self.assertTrue(np.all(pair.vals[:,4,1] == 4))
#
#         pair = Pair.cross_scalars(np.arange(12).reshape(3,4), np.arange(5))
#         self.assertEqual(pair.shape, [3,4,5])
#         self.assertTrue(np.all(pair.vals[2,3,:,0] == 11))
#         self.assertTrue(np.all(pair.vals[:,:,4,1] == 4))

        # New tests 2/1/12 (MRS)
        test = Pair(np.arange(6).reshape(3,2))
        self.assertEqual(str(test), "Pair([0 1]\n [2 3]\n [4 5])")

        test =  Pair(np.arange(6).reshape(3,2), mask=[False, False, True])
        self.assertEqual(str(test),   "Pair([0 1]\n [2 3]\n [-- --]; mask)")
        self.assertEqual(str(test*2), "Pair([0 2]\n [4 6]\n [-- --]; mask)")
        self.assertEqual(str(test/2), "Pair([0.0 0.5]\n [1.0 1.5]\n [-- --]; mask)")
        self.assertEqual(str(test%2), "Pair([0 1]\n [0 1]\n [-- --]; mask)")

        self.assertEqual(str(test + (1,0)),
                         "Pair([1 1]\n [3 3]\n [-- --]; mask)")
        self.assertEqual(str(test - (0,1)),
                         "Pair([0 0]\n [2 2]\n [-- --]; mask)")
        self.assertEqual(str(test + test),
                         "Pair([0 2]\n [4 6]\n [-- --]; mask)")
        self.assertEqual(str(test + np.arange(6).reshape(3,2)),
                         "Pair([0 2]\n [4 6]\n [-- --]; mask)")

        temp = Pair(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(test + temp),
                         "Pair([-- --]\n [4 6]\n [-- --]; mask)")
        self.assertEqual(str(test - 2*temp),
                         "Pair([-- --]\n [-2 -3]\n [-- --]; mask)")
        self.assertEqual(str(test.element_mul(temp)),
                         "Pair([-- --]\n [4 9]\n [-- --]; mask)")
        self.assertEqual(str(test.element_div(temp)),
                         "Pair([-- --]\n [1.0 1.0]\n [-- --]; mask)")

        temp = Pair(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(temp),      "Pair([-- --]\n [2 3]\n [4 5]; mask)")
        self.assertEqual(str(temp[0]),   "Pair(-- --; mask)")
        self.assertEqual(str(temp[1]),   "Pair(2 3)")
        self.assertEqual(str(temp[0:2]), "Pair([-- --]\n [2 3]; mask)")
        self.assertEqual(str(temp[0:1]), "Pair([-- --]; mask)")
        self.assertEqual(str(temp[1:2]), "Pair([2 3])")

        test = Pair(np.arange(6).reshape(3,2))
        self.assertEqual(test, Pair(np.arange(6).reshape(3,2)))

        mvals = test.mvals
        self.assertEqual(mvals.mask, np.ma.nomask)
        self.assertEqual(test, mvals)

        test = Pair(np.arange(6).reshape(3,2), [False, False, True])
        mvals = test.mvals
        self.assertEqual(str(mvals), "[[0 1]\n [2 3]\n [-- --]]")
        self.assertEqual(test.mask.shape, (3,))
        self.assertEqual(mvals.mask.shape, (3,2))
    #===========================================================================

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
