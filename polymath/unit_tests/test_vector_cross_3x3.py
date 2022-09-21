################################################################################
# Vector.cross() and Vector.cross_product_as_matrix() tests, 3x3 case
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

class Test_Vector_cross_3x3(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(9797)

    omega = Vector(np.random.randn(30,3))
    omega_as_matrix = omega.cross_product_as_matrix()

    vec = Vector(np.random.randn(20,30,3))

    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)

    self.assertTrue(np.all(np.abs(cross1.values - cross2.values) < 1.e-15))

    dots = omega.dot(cross1)
    self.assertTrue(np.all(np.abs(dots.values) < 1.e-14))

    # Test units
    omega = Vector(np.random.randn(3), units=Units.KM)
    omega_as_matrix = omega.cross_product_as_matrix()

    vec = Vector(np.random.randn(3), units=Units.SECONDS**(-1))

    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)

    self.assertEqual(cross1.units, Units.KM/Units.SECONDS)
    self.assertEqual(cross2.units, Units.KM/Units.SECONDS)

    # Derivatives, denom = ()
    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.cross(x)

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
    z1 = y.cross(x + (EPS,0,0))
    z0 = y.cross(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.cross(x + (0,EPS,0))
    z0 = y.cross(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.cross(x + (0,0,EPS))
    z0 = y.cross(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).cross(x)
    z0 = (y - (EPS,0,0)).cross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).cross(x)
    z0 = (y - (0,EPS,0)).cross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).cross(x)
    z0 = (y - (0,0,EPS)).cross(x)
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
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k], dz_df.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dg.values[i,k], dz_dg.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dh.values[i,k], dz_dh.values[i,k], delta=EPS)

    # Derivatives, denom = (3,), using matrix multiply
    z = y.cross_product_as_matrix() * x

    for i in range(N):
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k], dz_df.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dg.values[i,k], dz_dg.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dh.values[i,k], dz_dh.values[i,k], delta=EPS)

    # Derivatives, denom = (2,)
    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))

    z = y.cross(x)

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
    z1 = y.cross(x + (EPS,0,0))
    z0 = y.cross(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.cross(x + (0,EPS,0))
    z0 = y.cross(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.cross(x + (0,0,EPS))
    z0 = y.cross(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).cross(x)
    z0 = (y - (EPS,0,0)).cross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).cross(x)
    z0 = (y - (0,EPS,0)).cross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).cross(x)
    z0 = (y - (0,0,EPS)).cross(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS

    dz_df0 = (dz_dx0 * x.d_df.values[:,0,0] +
              dz_dx1 * x.d_df.values[:,1,0] +
              dz_dx2 * x.d_df.values[:,2,0])

    dz_df1 = (dz_dx0 * x.d_df.values[:,0,1] +
              dz_dx1 * x.d_df.values[:,1,1] +
              dz_dx2 * x.d_df.values[:,2,1])

    dz_dg0 = (dz_dy0 * y.d_dg.values[:,0,0] +
              dz_dy1 * y.d_dg.values[:,1,0] +
              dz_dy2 * y.d_dg.values[:,2,0])

    dz_dg1 = (dz_dy0 * y.d_dg.values[:,0,1] +
              dz_dy1 * y.d_dg.values[:,1,1] +
              dz_dy2 * y.d_dg.values[:,2,1])

    dz_dh0 = (dz_dx0 * x.d_dh.values[:,0,0] + dz_dy0 * y.d_dh.values[:,0,0] +
              dz_dx1 * x.d_dh.values[:,1,0] + dz_dy1 * y.d_dh.values[:,1,0] +
              dz_dx2 * x.d_dh.values[:,2,0] + dz_dy2 * y.d_dh.values[:,2,0])

    dz_dh1 = (dz_dx0 * x.d_dh.values[:,0,1] + dz_dy0 * y.d_dh.values[:,0,1] +
              dz_dx1 * x.d_dh.values[:,1,1] + dz_dy1 * y.d_dh.values[:,1,1] +
              dz_dx2 * x.d_dh.values[:,2,1] + dz_dy2 * y.d_dh.values[:,2,1])

    for i in range(N):
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k,0], dz_df0.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dg.values[i,k,0], dz_dg0.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dh.values[i,k,0], dz_dh0.values[i,k], delta=EPS)

        self.assertAlmostEqual(z.d_df.values[i,k,1], dz_df1.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dg.values[i,k,1], dz_dg1.values[i,k], delta=EPS)
        self.assertAlmostEqual(z.d_dh.values[i,k,1], dz_dh1.values[i,k], delta=EPS)

    # Derivatives should be removed if necessary
    self.assertEqual(y.cross(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.cross(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.cross(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.cross(x, recursive=False), 'd_dh'))

    # Read-only status should be preserved
    N = 10
    y = Vector(np.random.randn(N,3))
    x = Vector(np.random.randn(N,3))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.cross(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().cross(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().cross(x).readonly)
    self.assertFalse(y.cross(x.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
