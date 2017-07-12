################################################################################
# Vector.outer() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

class Test_Vector_outer(unittest.TestCase):

  def runTest(self):

    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,2))
    self.assertEquals(a.outer(b).shape, (3,10))
    self.assertEquals(a.outer(b).numer, (5,2))
    self.assertEquals(a.outer(b).denom, ())

    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    outer = a.outer(b)

    self.assertEqual(a.outer(b), a.values.reshape((1,10,5,1)) *
                                 b.values.reshape((3,10,1,5)))

    # Test units
    a = Vector(np.random.randn(3), units=Units.KM)
    b = Vector(np.random.randn(3), units=Units.SECONDS**(-1))

    self.assertEquals(a.outer(b).units, Units.KM/Units.SECONDS)
    self.assertEquals(b.outer(a).units, Units.KM/Units.SECONDS)

    # Derivatives
    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.outer(x)

    self.assertIn('f', x.derivs)
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertNotIn('g', x.derivs)
    self.assertFalse(hasattr(x, 'd_dg'))
    self.assertIn('h', x.derivs)
    self.assertTrue(hasattr(x, 'd_dh'))

    self.assertNotIn('f', y.derivs)
    self.assertFalse(hasattr(y, 'd_df'))
    self.assertIn('g', y.derivs)
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertIn('h', y.derivs)
    self.assertTrue(hasattr(y, 'd_dh'))

    self.assertIn('f', z.derivs)
    self.assertTrue(hasattr(z, 'd_df'))
    self.assertIn('g', z.derivs)
    self.assertTrue(hasattr(z, 'd_dg'))
    self.assertIn('h', z.derivs)
    self.assertTrue(hasattr(z, 'd_dh'))

    EPS = 1.e-6
    z1 = y.outer(x + (EPS,0,0))
    z0 = y.outer(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.outer(x + (0,EPS,0))
    z0 = y.outer(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.outer(x + (0,0,EPS))
    z0 = y.outer(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).outer(x)
    z0 = (y - (EPS,0,0)).outer(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).outer(x)
    z0 = (y - (0,EPS,0)).outer(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).outer(x)
    z0 = (y - (0,0,EPS)).outer(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS

    dz_df = (dz_dx0 * x.d_df.values[:,0] +
             dz_dx1 * x.d_df.values[:,1] +
             dz_dx2 * x.d_df.values[:,2])

    dz_dg = (dz_dy0 * y.d_dg.values[:,0] +
             dz_dy1 * y.d_dg.values[:,1] +
             dz_dy2 * y.d_dg.values[:,2])

    dz_dh = (dz_dx0 * x.d_dh.values[:,0] + dz_dy0 * y.d_dh.values[:,0] +
             dz_dx1 * x.d_dh.values[:,1] + dz_dy1 * y.d_dh.values[:,1] +
             dz_dx2 * x.d_dh.values[:,2] + dz_dy2 * y.d_dh.values[:,2])

    for i in range(N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(z.d_df.values[i,j,k], dz_df.values[i,j,k],
                                                         delta=EPS)
            self.assertAlmostEqual(z.d_dg.values[i,j,k], dz_dg.values[i,j,k],
                                                         delta=EPS)
            self.assertAlmostEqual(z.d_dh.values[i,j,k], dz_dh.values[i,j,k],
                                                         delta=EPS)

    # Derivatives should be removed if necessary
    self.assertEqual(y.outer(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.outer(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.outer(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.outer(x, recursive=False), 'd_dh'))

    # Read-only status should be preserved
    N = 10
    y = Vector(np.random.randn(N,7))
    x = Vector(np.random.randn(N,7))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.outer(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().outer(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().outer(x).readonly)
    self.assertFalse(y.outer(x.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
