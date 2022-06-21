################################################################################
# Old Matrix tests, updated by MRS 2/19/14
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Matrix, Scalar, Units

#*******************************************************************************
# Test_Matrix_misc
#*******************************************************************************
class Test_Matrix_misc(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        a = Vector((1,2))
        b = Vector((0,1,-1))

        #--------------------
        # Outer multiply     	     
        #--------------------
        ab = a.outer(b)

        self.assertEqual(ab, Matrix([(0.,1.,-1.),
                                      (0.,2.,-2.)]))

        self.assertEqual(ab * Vector((3,2,1)), Vector([1.,2.]))
        self.assertEqual(ab * Vector([(3,2,1),
                                      (1,2,0)]), Vector(([1.,2.],
                                                         [2.,4.])))

        v = Vector([(3,2,1),(1,2,0)])
        self.assertEqual(v.shape, (2,))
        self.assertEqual(v.item, (3,))
        self.assertEqual(v*2, Vector([(6,4,2),(2,4,0)]))
        self.assertEqual(v/2, Vector([(1.5,1.,0.5),(0.5,1.,0.)]))
        self.assertEqual(2*v, 2.*v)

        m = Matrix([(3,2,1),(1,2,0)])
        self.assertEqual(m.shape, ())
        self.assertEqual(m.item, (2,3))
        self.assertEqual(m*2, Matrix([(6,4,2),(2,4,0)]))
        self.assertEqual(m/2, Matrix([(1.5,1.,0.5),(0.5,1.,0.)]))
        self.assertEqual(2*m, 2.*m)

        i = Matrix([(-1,0,0),(0,2,0),(0,0,0)])
        self.assertEqual(m*i, Matrix([(-3,4,0),(-1,4,0)]))
        self.assertEqual(i*v, Vector([(-3,4,0),(-1,4,0)]))

        j = Matrix([(-1,0),(0,2),(1,1)])
        self.assertEqual(j*m, Matrix([(-3,-2,-1),(2,4,0),(4,4,1)]))

        #------------------------
        # 3x3 Matrix inverse	 	 
        #------------------------
        test = Matrix(np.random.rand(200,3,3))
        inverse = test.inverse()
        product = test * inverse

        DEL = 1.e-11
        self.assertTrue(np.all(abs(product.vals[...,0,0] - 1) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,1,1] - 1) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,2,2] - 1) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,0,1]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,1,0]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,2,0]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,0,2]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,2,1]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,1,2]) < DEL))
  #=============================================================================



#*******************************************************************************



############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
