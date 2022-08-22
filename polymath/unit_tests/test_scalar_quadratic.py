################################################################################
# Scalar.solve_quadratic() and Scalar.eval_quadratic() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Scalar, Vector3

#*******************************************************************************
# Test_Scalar_quadratic
#*******************************************************************************
class Test_Scalar_quadratic(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-----------------------------
    # Arrays of various sizes
    #-----------------------------
    a = np.random.randn(8)
    b = np.random.randn(3,8)
    c = np.random.randn(4,1,1)

    (x0, x1) = Scalar.solve_quadratic(a, b, c)

    self.assertEqual(x0.shape, (4,3,8))
    self.assertTrue(abs(x0.eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x0.eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(abs(x1.eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x1.eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(np.all(x0.mask == x1.mask))

    a = np.random.randn(8)
    b = np.random.randn(3,8)
    c = np.random.randn(4,1,1)

    #----------------------------------------------------------
    # Check case where sometimes there is only one solution
    #----------------------------------------------------------
    a[0] = 0.
    (x0, x1) = Scalar.solve_quadratic(a, b, c)
    self.assertTrue(abs(x0.eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x0.eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(abs(x1.eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x1.eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(np.all(x0[...,1:].mask == x1[...,1:].mask))
    self.assertTrue(np.all(x1[...,0].mask))

    #---------------------
    # Single values
    #---------------------
    for k in range(100):
        a = np.random.randn()
        b = np.random.randn()
        c = np.random.randn()

        (x0, x1) = Scalar.solve_quadratic(a, b, c)

        self.assertEqual(x0.shape, ())
        if not x0.mask:
            self.assertTrue(x0.eval_quadratic(a,b,c) < 3.e-13)
            self.assertTrue(x1.eval_quadratic(a,b,c) < 3.e-13)
            self.assertTrue(x0.mask == x1.mask)

    #------------------------
    # Single linear case
    #------------------------
    a = 0.
    b = np.random.randn()
    c = np.random.randn()

    (x0, x1) = Scalar.solve_quadratic(a, b, c)
    self.assertTrue(x0.eval_quadratic(a,b,c) < 3.e-13)
    self.assertTrue(x1.mask)

    #-----------------------
    # Derivatives wrt a
    #-----------------------
    a = Scalar(np.random.randn(8))
    b = Scalar(np.random.randn(3,8))
    c = Scalar(np.random.randn(4,1,1))

    a.insert_deriv('t', Scalar(np.random.randn(8)))

    x = Scalar.solve_quadratic(a, b, c)
    self.assertTrue(abs(x[0].eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x[0].eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(abs(x[1].eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x[1].eval_quadratic(a,b,c)).max()    < 1.e-13)
    self.assertTrue('t' in x[0].derivs)
    self.assertTrue('t' in x[1].derivs)

    da = 1.e-5 * a
    for k in range(2):
        dx = 0.5 * (Scalar.solve_quadratic(a + da, b, c)[k] -
                    Scalar.solve_quadratic(a - da, b, c)[k])
        self.assertTrue(abs(dx * a.d_dt - x[k].d_dt * da).median() < 3.e-14)

    #------------------------
    # Derivatives wrt b
    #------------------------
    a = Scalar(np.random.randn(8))
    b = Scalar(np.random.randn(3,8))
    c = Scalar(np.random.randn(4,1,1))

    b.insert_deriv('t', Scalar(np.random.randn(3,8)))

    x = Scalar.solve_quadratic(a, b, c)
    self.assertTrue(abs(x[0].eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x[0].eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(abs(x[1].eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x[1].eval_quadratic(a,b,c)).max()    < 3.e-13)
    self.assertTrue('t' in x[0].derivs)
    self.assertTrue('t' in x[1].derivs)

    db = 1.e-5 * b
    for k in range(2):
        dx = 0.5 * (Scalar.solve_quadratic(a, b+db, c)[k] -
                    Scalar.solve_quadratic(a, b-db, c)[k])
        self.assertTrue(abs(dx * b.d_dt - x[k].d_dt * db).median() < 3.e-14)

    #-------------------------
    # Derivatives wrt c
    #-------------------------
    a = Scalar(np.random.randn(8))
    b = Scalar(np.random.randn(3,8))
    c = Scalar(np.random.randn(4,1,1))
    c.insert_deriv('t', Scalar(np.random.randn(4,1,1)))

    x = Scalar.solve_quadratic(a, b, c)
    self.assertTrue(abs(x[0].eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x[0].eval_quadratic(a,b,c)).max()    < 3.e-13)

    self.assertTrue(abs(x[1].eval_quadratic(a,b,c)).median() < 1.e-15)
    self.assertTrue(abs(x[1].eval_quadratic(a,b,c)).max()    < 3.e-13)
    self.assertTrue('t' in x[0].derivs)
    self.assertTrue('t' in x[1].derivs)

    dc = 1.e-5 * c
    for k in range(2):
        dx = 0.5 * (Scalar.solve_quadratic(a, b, c+dc)[k] -
                    Scalar.solve_quadratic(a, b, c-dc)[k])
        self.assertTrue(abs(dx * c.d_dt - x[k].d_dt * dc).median() < 1.e-14)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
