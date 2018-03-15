################################################################################
# Vector.as_diagonal() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Vector, Scalar, Units

class Test_Vector_as_diagonal(unittest.TestCase):

  def runTest(self):

    # Check one matrix
    a = Vector(np.arange(6))
    b = a.as_diagonal()
    for i in range(6):
        for j in range(6):
            if i == j:
                self.assertEqual(b.values[i,i], a.values[i])
            else:
                self.assertEqual(b.values[i,j], 0.)

    # Check an array of matrices, some masked
    N = 10
    a = Vector(np.random.randn(100,4), mask= np.random.rand(100) < -0.05)
    b = a.as_diagonal()

    for i in range(4):
      for j in range(4):
        aa = a.extract_numer(0, i, Scalar)
        bb = b.extract_numer(0, i, Vector).extract_numer(0, j, Scalar)

        if i == j:
            self.assertEqual(bb, aa)
        else:
            self.assertEqual(bb, 0.)

    self.assertTrue(np.all(a.mask == b.mask))

    # Test units
    a = Vector(np.random.randn(4), units=Units.KM)

    self.assertEqual(a.as_diagonal().units, Units.KM)

    # Derivatives
    N = 100
    x = Vector(np.random.randn(N,3))

    x.insert_deriv('t', Vector(np.random.randn(N,3)))
    x.insert_deriv('v', Vector(np.random.randn(N,3,2), drank=1))
    y = x.as_diagonal()

    self.assertIn('t', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertIn('v', x.derivs)
    self.assertTrue(hasattr(x, 'd_dv'))

    self.assertIn('t', y.derivs)
    self.assertTrue(hasattr(y, 'd_dt'))
    self.assertIn('v', y.derivs)
    self.assertTrue(hasattr(y, 'd_dv'))

    EPS = 1.e-6
    y1 = (x + (EPS,0,0)).as_diagonal()
    y0 = (x - (EPS,0,0)).as_diagonal()
    dy_dx0 = 0.5 * (y1 - y0) / EPS

    y1 = (x + (0,EPS,0)).as_diagonal()
    y0 = (x - (0,EPS,0)).as_diagonal()
    dy_dx1 = 0.5 * (y1 - y0) / EPS

    y1 = (x + (0,0,EPS)).as_diagonal()
    y0 = (x - (0,0,EPS)).as_diagonal()
    dy_dx2 = 0.5 * (y1 - y0) / EPS

    new_values = np.empty((N,3,3,3))
    new_values[...,0] = dy_dx0.values
    new_values[...,1] = dy_dx1.values
    new_values[...,2] = dy_dx2.values

    dy_dx = Matrix(new_values, drank=1)

    dy_dt = dy_dx.chain(x.d_dt)
    dy_dv = dy_dx.chain(x.d_dv)

    DEL = 1.e-5
    for i in range(N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(dy_dt.values[i,j,k],
                                   y.d_dt.values[i,j,k], delta=DEL)
            self.assertAlmostEqual(dy_dv.values[i,j,k,0],
                                   y.d_dv.values[i,j,k,0], delta=DEL)
            self.assertAlmostEqual(dy_dv.values[i,j,k,1],
                                   y.d_dv.values[i,j,k,1], delta=DEL)

    # Derivatives should be removed if necessary
    self.assertEqual(x.as_diagonal(recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dv'))
    self.assertFalse(hasattr(x.as_diagonal(recursive=False), 'd_dt'))
    self.assertFalse(hasattr(x.as_diagonal(recursive=False), 'd_dv'))

    # Read-only status should NOT be preserved
    N = 10
    x = Vector(np.random.randn(N,7))

    self.assertFalse(x.readonly)
    self.assertFalse(x.as_diagonal().readonly)
    self.assertFalse(x.as_readonly().as_diagonal().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
