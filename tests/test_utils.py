################################################################################
# tests/test_utils.py
################################################################################

import numpy as np
import unittest

from oops.utils import *


class Test_Utils(unittest.TestCase):

    def runTest(self):

        np.random.seed(6167)

        # dot
        self.assertEqual(dot((1,2),(3,4)), 11)
        self.assertEqual(dot((1,2,3),(3,4,5)), 26)
        self.assertEqual(dot((1.,2.),(3.,4.)), 11.)
        self.assertEqual(dot((1.,2.,3.),(3.,4.,5.)), 26.)
        self.assertTrue(np.all(dot([(1.,2.),(-1.,-2.)],(3.,4.)) == [11.,-11]))
        self.assertTrue(np.all(dot([(1.,2.,3.),(3.,2.,1.)],(3.,4.,5.))
                                   == (26.,22.)))

        # norm
        self.assertEqual(norm((3,4)), 5.)
        self.assertEqual(norm((3,4,12)), 13.)
        self.assertTrue(np.all(norm([(3,4),(5,12)]) == [5.,13.]))
        self.assertTrue(np.all(norm([(3,4,12),(5,12,84)]) == [13.,85.]))

        # unit, sep
        eps = 3.e-16
        lo = 1. - eps
        hi = 1. + eps
        self.assertTrue(norm(unit((3,4))) > lo)
        self.assertTrue(norm(unit((3,4))) < hi)

        test2 = [[(1,2),(3,4)],[(5,6),(7,8)]]
        self.assertTrue(np.all(norm(unit(test2)) > lo))
        self.assertTrue(np.all(norm(unit(test2)) < hi))
        self.assertTrue(np.all(norm(unit(test2)) > [lo,lo]))
        self.assertTrue(np.all(norm(unit(test2)) < [hi,hi]))
        self.assertTrue(np.all(norm(unit(test2)) > [[lo,lo],[lo,lo]]))
        self.assertTrue(np.all(norm(unit(test2)) < [[hi,hi],[hi,hi]]))

        self.assertTrue(np.all(sep(test2,unit(test2)) <  eps))
        self.assertTrue(np.all(sep(test2,unit(test2)) > -eps))
        self.assertTrue(np.all(sep(test2,unit(test2)) < [ eps, eps]))
        self.assertTrue(np.all(sep(test2,unit(test2)) > [-eps,-eps]))

        self.assertTrue(norm(unit((3,4,5))) > lo)
        self.assertTrue(norm(unit((3,4,5))) < hi)

        test3 = [[(1,2,-3),(3,4,-5)],[(5,6,-7),(7,8,-9)]]
        self.assertTrue(np.all(norm(unit(test3)) > lo))
        self.assertTrue(np.all(norm(unit(test3)) < hi))
        self.assertTrue(np.all(norm(unit(test3)) > [lo,lo]))
        self.assertTrue(np.all(norm(unit(test3)) < [hi,hi]))
        self.assertTrue(np.all(norm(unit(test3)) > [[lo,lo],[lo,lo]]))
        self.assertTrue(np.all(norm(unit(test3)) < [[hi,hi],[hi,hi]]))

        self.assertTrue(np.all(sep(test3,unit(test3)) <  eps))
        self.assertTrue(np.all(sep(test3,unit(test3)) > -eps))
        self.assertTrue(np.all(sep(test3,unit(test3)) < [ eps, eps]))
        self.assertTrue(np.all(sep(test3,unit(test3)) > [-eps,-eps]))

        # cross2d, sep
        self.assertEqual(cross2d((1,0),(0,1)), 1.)
        self.assertEqual(cross2d((1,0),(1,1)), 1.)
        self.assertEqual(cross2d((1,0),(111,1)), 1.)
        self.assertEqual(cross2d((0,1),(111,1)), -111.)

        dirs = np.asfarray([[[( 5, 0),( 4, 3),( 3, 4)],
                             [( 0, 5),(-3, 4),(-4, 3)]],
                            [[(-5, 0),(-4,-3),(-3,-4)],
                             [( 0,-5),( 3,-4),( 4,-3)]]])
        self.assertTrue(np.all(cross2d(dirs,(1,0)) == -dirs[...,1]))
        self.assertTrue(np.all(cross2d(dirs,(0,1)) ==  dirs[...,0]))

        # cross3d
        self.assertTrue(np.all(cross3d((1,0,0),(0,1,0)) == (0, 0,1)))
        self.assertTrue(np.all(cross3d((1,0,0),(0,0,1)) == (0,-1,0)))

        self.assertTrue(np.all(cross3d([(1 ,0,0),(0,2,0)],(0,0,1)) ==
                                       [(0,-1,0),(2,0,0)]))

        # ucross3d, sep, norm
        eps = 1.e-15
        vec1 = [(7,-1,1),(1,2,-3),(-1,3,3)]
        vec2 = (3,1,-3)

        test = ucross3d(vec1, vec2)
        self.assertTrue(np.all(norm(test) > 1. - eps))
        self.assertTrue(np.all(norm(test) < 1. + eps))
        self.assertTrue(np.all(sep(test,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(test,vec2) < np.pi/2. + eps))

        vec2 = [(3,2,1),(-4,-1,0),(7,6,5)]

        test = ucross3d(vec1, vec2)
        self.assertTrue(np.all(norm(test) > 1. - eps))
        self.assertTrue(np.all(norm(test) < 1. + eps))
        self.assertTrue(np.all(sep(test,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(test,vec2) < np.pi/2. + eps))

        # proj, perp, sep, norm
        eps = 3.e-15
        perps = perp(vec1, vec2)
        projs = proj(vec1, vec2)

        self.assertTrue(np.all(dot(perps,vec2) > -eps))
        self.assertTrue(np.all(dot(perps,vec2) <  eps))

        self.assertTrue(np.all(sep(perps,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(perps,vec2) < np.pi/2. + eps))

        self.assertTrue(np.all(sep(projs,vec2) % np.pi > -eps))
        self.assertTrue(np.all(sep(projs,vec2) % np.pi <  eps))

        test = vec1 - (projs + perps)
        self.assertTrue(np.all(test > -eps))
        self.assertTrue(np.all(test <  eps))

        vec2 = [(3,2,1),(-4,-1,0),(7,6,5)]

        self.assertTrue(np.all(dot(perps,vec2) > -eps))
        self.assertTrue(np.all(dot(perps,vec2) <  eps))

        self.assertTrue(np.all(sep(perps,vec2) > np.pi/2. - eps))
        self.assertTrue(np.all(sep(perps,vec2) < np.pi/2. + eps))

        self.assertTrue(np.all(sep(projs,vec2) % np.pi > -eps))
        self.assertTrue(np.all(sep(projs,vec2) % np.pi <  eps))

        test = vec1 - (projs + perps)
        self.assertTrue(np.all(test > -eps))
        self.assertTrue(np.all(test <  eps))

        # xpose
        mat = [[[1,2,3],[4,5,6],[7,8,9]]] * 7
        self.assertEqual(np.shape(mat),(7,3,3))
        self.assertEqual(np.shape(xpose(mat)),(7,3,3))
        self.assertTrue(np.all(np.array(mat)[...,0,1] == xpose(mat)[...,1,0]))

        # twovec, mxv, mtxv, twovec
        eps = 1.e-14

        mat1 = twovec((1,0,0),0,(0,1,0),1)
        mat2 = twovec((1,0,0),0,(0,0,4),2)
        self.assertTrue(np.all(mat1 == mat2))
        self.assertTrue(np.all(mat1 == [[1,0,0,],[0,1,0],[0,0,1]]))

        self.assertTrue(np.all(mxv( mat1,vec1) == vec1))
        self.assertTrue(np.all(mtxv(mat1,vec1) == vec1))
        self.assertTrue(np.all(mxv( mat1,vec1[0]) == vec1[0]))
        self.assertTrue(np.all(mtxv(mat1,vec1[0]) == vec1[0]))

        # Rotate vectors along the axes into the frame
        mat = twovec((1,1,1),2,[(1,0,-1),(-1,0,1)],0)
        vec = (3,3,3)

        self.assertTrue(np.all(mxv(mat,vec)[...,0:2] > -eps))
        self.assertTrue(np.all(mxv(mat,vec)[...,0:2] <  eps))
        self.assertTrue(np.all(mxv(mat,vec)[...,2] > np.sqrt(27) - eps))
        self.assertTrue(np.all(mxv(mat,vec)[...,2] < np.sqrt(27) + eps))

        vec = [(2,0,-2),[-2,0,2]]
        result = self.assertTrue(np.all(mxv(mat,vec)[:,1:3] > -eps))
        result = self.assertTrue(np.all(mxv(mat,vec)[:,1:3] <  eps))
        result = self.assertTrue(np.all(mxv(mat,vec)[:,0] > np.sqrt(8) - eps))
        result = self.assertTrue(np.all(mxv(mat,vec)[:,0] < np.sqrt(8) + eps))

        # Rotate axis vectors out of the frame
        vec = [[(1,0,0),[0,1,0]],[(2,3,4),[0,0,2]]]
        result = mtxv(mat,vec)

        self.assertEqual(result[1,1,0], result[1,1,1])
        self.assertEqual(result[1,1,0], result[1,1,2])
        self.assertEqual(result[0,0,0],-result[0,0,2])
        self.assertEqual(result[0,0,1], 0.)

        result = mxv(xpose(mat),vec)
        self.assertEqual(result[1,1,0], result[1,1,1])
        self.assertEqual(result[1,1,0], result[1,1,2])
        self.assertEqual(result[0,0,0],-result[0,0,2])
        self.assertEqual(result[0,0,1], 0.)

        mat = [[1,2,3],[4,5,6],[7,8,9]]
        vec = [1,0,0]
        self.assertTrue(np.all(mxv(mat,vec)  - [1,4,7]) == 0.)
        self.assertTrue(np.all(mtxv(mat,vec) - [1,2,3]) == 0.)
        vec = [0,1,0]
        self.assertTrue(np.all(mxv(mat,vec)  - [2,5,8]) == 0.)
        self.assertTrue(np.all(mtxv(mat,vec) - [4,5,6]) == 0.)

        # mxv, mtxv, mxm, mtxm, mxmt, mtxmt, with shape broadcasting
        a = np.random.rand(2,1,4,3,3)
        b = np.random.rand(  3,4,3,3)
        v = np.random.rand(1,3,1,3,1)

        axb   = mxm(a,b)
        atxb  = mtxm(a,b)
        axbt  = mxmt(a,b)
        atxbt = mtxmt(a,b)

        axv  = mxv(a,v[...,0])
        atxv = mtxv(a,v[...,0])

        self.assertEqual(axb.shape, (2,3,4,3,3))
        self.assertEqual(axv.shape, (2,3,4,3))

        eps = 1.e-15

        for i in range(2):
          for j in range(3):
            for k in range(4):
                am = np.array(a[i,0,k])
                bm = np.array(b[  j,k])
                amt = np.array(a[i,0,k].T)
                bmt = np.array(b[  j,k].T)
                vm  = np.array(v[0,j,0])

                test = am @ bm
                self.assertLess(np.abs(test - axb[i,j,k]).max(), eps)

                test = amt @ bm
                self.assertLess(np.abs(test - atxb[i,j,k]).max(), eps)

                test = am @ bmt
                self.assertLess(np.abs(test - axbt[i,j,k]).max(), eps)

                test = amt @ bmt
                self.assertLess(np.abs(test - atxbt[i,j,k]).max(), eps)

                test = am @ vm
                self.assertLess(np.abs(test - axv[i,j,k,:,np.newaxis]).max(), eps)

                test = amt @ vm
                self.assertLess(np.abs(test - atxv[i,j,k,:,np.newaxis]).max(), eps)

########################################
if __name__ == "__main__":
    unittest.main(verbosity=2)
################################################################################
