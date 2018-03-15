################################################################################
# Vector.dot() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

class Test_Vector_dot(unittest.TestCase):

  def runTest(self):

    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,4))
    self.assertRaises(ValueError, a.dot, b)

    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    dot = a.dot(b)

    self.assertEqual(a.dot(b), np.sum(a.values * b.values, axis=-1))

    # Test units
    omega = Vector(np.random.randn(3), units=Units.KM)
    omega_as_matrix = omega.cross_product_as_matrix()

    vec = Vector(np.random.randn(3), units=Units.SECONDS**(-1))

    cross1 = omega_as_matrix * vec
    cross2 = omega.dot(vec)

    self.assertEqual(cross1.units, Units.KM/Units.SECONDS)
    self.assertEqual(cross2.units, Units.KM/Units.SECONDS)

    # Derivatives
    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.dot(x)

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
    z1 = y.dot(x + (EPS,0,0))
    z0 = y.dot(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.dot(x + (0,EPS,0))
    z0 = y.dot(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.dot(x + (0,0,EPS))
    z0 = y.dot(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).dot(x)
    z0 = (y - (EPS,0,0)).dot(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).dot(x)
    z0 = (y - (0,EPS,0)).dot(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).dot(x)
    z0 = (y - (0,0,EPS)).dot(x)
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
        self.assertAlmostEqual(z.d_df.values[i], dz_df.values[i], delta=EPS)
        self.assertAlmostEqual(z.d_dg.values[i], dz_dg.values[i], delta=EPS)
        self.assertAlmostEqual(z.d_dh.values[i], dz_dh.values[i], delta=EPS)

    # Derivatives should be removed if necessary
    self.assertEqual(y.dot(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.dot(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.dot(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.dot(x, recursive=False), 'd_dh'))

    # Read-only status should NOT be preserved
    N = 10
    y = Vector(np.random.randn(N,7))
    x = Vector(np.random.randn(N,7))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.dot(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().dot(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().dot(x).readonly)
    self.assertFalse(y.dot(x.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
