################################################################################
# Old Vector3 tests, updated by MRS 2/18/14
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Boolean, Scalar, Vector, Vector3, Pair, Units

class Test_Vector3_misc(unittest.TestCase):

  # runTest
    def runTest(self):

        np.random.seed(2222)

        eps = 3.e-16
        lo = 1. - eps
        hi = 1. + eps

        # Basic comparisons and indexing
        vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])
        self.assertEqual(vecs.numer, (3,))
        self.assertEqual(vecs.shape, (3,))
        self.assertEqual(vecs.rank,   1)

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
        self.assertEqual((vecs == test), Boolean(True))
        self.assertEqual((vecs != test), Boolean(False))
        self.assertEqual((vecs == test), Boolean((True,  True,  True)))
        self.assertEqual((vecs != test), Boolean((False, False, False)))

        self.assertEqual((vecs == [1,2,3]), Boolean((True, False, False)))

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
        vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])

        self.assertEqual(vecs + (0,1,2), [[1,3,5],[3,5,7],(5,7,9)])
        self.assertEqual(vecs + (0,1,2), Vector3([[1,3,5],[3,5,7],(5,7,9)]))
        self.assertEqual(vecs - (0,1,2), [[1,1,1],[3,3,3],[5,5,5]])
        self.assertEqual(vecs - (0,1,2), Vector3([[1,1,1],[3,3,3],[5,5,5]]))

        self.assertEqual(vecs.element_mul((1,2,3)),
                                [[1,4,9],[3,8,15],[5,12,21]])
        self.assertEqual(vecs.element_mul((1,2,3)),
                                Vector3([[1,4,9],[3,8,15],[5,12,21]]))
        self.assertEqual(vecs.element_mul(Vector3((1,2,3))),
                                [[1,4,9],[3,8,15],[5,12,21]])
        self.assertEqual(vecs.element_mul(Vector3((1,2,3))),
                                Vector3([[1,4,9],[3,8,15],[5,12,21]]))

        self.assertEqual(vecs * 2, [[2,4,6],[6,8,10],[10,12,14]])
        self.assertEqual(vecs * 2, Vector3([[2,4,6],[6,8,10],[10,12,14]]))
        self.assertEqual(vecs * Scalar(2), [[2,4,6],[6,8,10],[10,12,14]])
        self.assertEqual(vecs * Scalar(2), Vector3([[2,4,6],[6,8,10],
                                                                 [10,12,14]]))

        self.assertEqual(vecs.element_div((1,1,2)),
                                [[1,2,1.5],[3,4,2.5],[5,6,3.5]])
        self.assertEqual(vecs.element_div(Vector3((1,1,2))),
                                [[1,2,1.5],[3,4,2.5],[5,6,3.5]])

        self.assertEqual(vecs / 2, [[0.5,1,1.5],[1.5,2,2.5],[2.5,3,3.5]])
        self.assertEqual(vecs / Scalar(2), [[0.5,1,1.5],[1.5,2,2.5],
                                                             [2.5,3,3.5]])

        self.assertRaises(TypeError, vecs.__add__, 1)
        self.assertRaises(TypeError, vecs.__add__, Scalar(1))
        self.assertRaises(ValueError, vecs.__add__, (1,2))
        self.assertRaises(TypeError, vecs.__add__, Pair((1,2)))

        self.assertRaises(TypeError, vecs.__sub__, 1)
        self.assertRaises(TypeError, vecs.__sub__, Scalar(1))
        self.assertRaises(ValueError, vecs.__sub__, (1,2))
        self.assertRaises(TypeError, vecs.__sub__, Pair((1,2)))

        self.assertRaises(ValueError, vecs.__mul__, (1,2))
        self.assertRaises(TypeError, vecs.__mul__, Pair((1,2)))

        self.assertRaises(ValueError, vecs.__div__, (1,2))
        self.assertRaises(TypeError, vecs.__div__, Pair((1,2)))

        # In-place operations
        vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])
        test = vecs.copy()
        test += (1,2,3)
        self.assertEqual(test, [[2,4,6],[4,6,8],(6,8,10)])
        test -= (1,2,3)
        self.assertEqual(test, vecs)
        test = test.element_mul((1,2,3))
        self.assertEqual(test, [[1,4,9],[3,8,15],[5,12,21]])
        test = test.element_div((1,2,3))
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

        self.assertRaises(TypeError, test.__iadd__, Scalar(1))
        self.assertRaises(TypeError, test.__iadd__, 1)
        self.assertRaises(ValueError, test.__iadd__, (1,2))

        self.assertRaises(TypeError, test.__isub__, Scalar(1))
        self.assertRaises(TypeError, test.__isub__, 1)
        self.assertRaises(ValueError, test.__isub__, (1,2,3,4))

        self.assertRaises(TypeError, test.__imul__, Pair((1,2)))
        self.assertRaises(ValueError, test.__imul__, (1,2,3,4))

        self.assertRaises(TypeError, test.__idiv__, Pair((1,2)))
        self.assertRaises(ValueError, test.__idiv__, (1,2,3,4))

        # Other functions...

        # to_scalar()
        self.assertEqual(vecs.to_scalar(0),  Scalar((1,3,5)))
        self.assertEqual(vecs.to_scalar(1),  Scalar((2,4,6)))
        self.assertEqual(vecs.to_scalar(2),  Scalar((3,5,7)))
        self.assertEqual(vecs.to_scalar(-1), Scalar((3,5,7)))
        self.assertEqual(vecs.to_scalar(-2), Scalar((2,4,6)))
        self.assertEqual(vecs.to_scalar(-3), Scalar((1,3,5)))

        # to_scalars()
        self.assertEqual(vecs.to_scalars(), (Scalar((1,3,5)),
                                             Scalar((2,4,6)),
                                             Scalar((3,5,7))))

        # dot()
        self.assertEqual(vecs.dot((1,0,0)), vecs.to_scalar(0))
        self.assertEqual(vecs.dot((0,1,0)), vecs.to_scalar(1))
        self.assertEqual(vecs.dot((0,0,1)), vecs.to_scalar(2))
        self.assertEqual(vecs.dot((1,1,0)),
                         vecs.to_scalar(0) + vecs.to_scalar(1))

        # norm()
        v = Vector3([[[1,2,3],[2,3,4]],[[0,1,2],[3,4,5]]])
        self.assertEqual(v.norm(), np.sqrt([[14,29],[5,50]]))

        # cross(), ucross()
        a = Vector3([[[1,0,0]],[[0,2,0]],[[0,0,3]]])
        b = Vector3([ [0,3,3] , [2,0,2] , [1,1,0] ])
        axb = a.cross(b)

        self.assertEqual(a.shape,   (3,1))
        self.assertEqual(b.shape,    (3,))
        self.assertEqual(axb.shape, (3,3))

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

        self.assertEqual(aperp.shape, (2,3,4,2))
        self.assertEqual(aproj.shape, (2,3,4,2))

        eps = 3.e-14
        self.assertTrue((aperp.sep(b) > np.pi/2 - eps).all())
        self.assertTrue((aperp.sep(b) < np.pi/2 + eps).all())
        self.assertTrue((aproj.sep(b) % np.pi > -eps).all())
        self.assertTrue((aproj.sep(b) % np.pi <  eps).all())
        self.assertTrue(np.all((a - aperp - aproj).vals > -eps))
        self.assertTrue(np.all((a - aperp - aproj).vals <  eps))

        # Note: the sep(reverse=True) option is not tested here

        # New tests 2/1/12 (MRS)
        test = Vector3(np.arange(6).reshape(2,3))
        str_test = str(test).replace('  ', ' ').replace('[ ','[')
        self.assertEqual(str_test, "Vector3([0. 1. 2.]\n [3. 4. 5.])")

        test = Vector3(np.arange(6).reshape(2,3), mask = [True, False])
        self.assertEqual(str(test),
                "Vector3([-- -- --]\n [3.0 4.0 5.0]; mask)")
        self.assertEqual(str(test*2),
                "Vector3([-- -- --]\n [6.0 8.0 10.0]; mask)")
        self.assertEqual(str(test/2),
                "Vector3([-- -- --]\n [1.5 2.0 2.5]; mask)")

        self.assertEqual(str(test + (1,0,2)),
                "Vector3([-- -- --]\n [4.0 4.0 7.0]; mask)")
        self.assertEqual(str(test - (1,0,2)),
                "Vector3([-- -- --]\n [2.0 4.0 3.0]; mask)")
        self.assertEqual(str(test - 2*test),
                "Vector3([-- -- --]\n [-3.0 -4.0 -5.0]; mask)")
        self.assertEqual(str(test + np.arange(6).reshape(2,3)),
                "Vector3([-- -- --]\n [6.0 8.0 10.0]; mask)")

        self.assertEqual(str(test[0]),
                "Vector3(-- -- --; mask)")
        self.assertEqual(str(test[1]).replace('( ','(').replace('  ',' '),
                "Vector3(3. 4. 5.)")
        self.assertEqual(str(test[0:2]),
                "Vector3([-- -- --]\n [3.0 4.0 5.0]; mask)")
        self.assertEqual(str(test[0:1]),
                "Vector3([-- -- --]; mask)")
        self.assertEqual(str(test[1:2]).replace('[ ','[').replace('  ',' '),
                "Vector3([3. 4. 5.])")

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
