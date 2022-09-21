################################################################################
# Vector.perp()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

class Test_Vector_perp(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(2435)

    # Single values
    self.assertEqual(Vector((2,3,0)).perp((0,7,0)), (2,0,0))
    self.assertEqual(Vector((2,3,0)).perp((-1,0,0)), (0,3,0))
    self.assertTrue(Vector((2,3,0),True).perp((-1,0,0)).mask)
    self.assertTrue(Vector((2,3,0)).perp((0,0,0)).mask)
    self.assertEqual(Vector((0,0,0)).perp((1,1,1)).norm(), 0.)

    # Arrays and masks
    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    z = y.perp(x)

    for i in range(N):
        self.assertAlmostEqual(z[i].cross(x[i]).norm(),
                               z[i].norm() * x[i].norm(), delta=1.e-14)
        self.assertAlmostEqual(z[i].dot(x[i]), 0., delta=1.e-14)

    N = 100
    x = Vector(np.random.randn(N,3), np.random.randn(N) < -0.5)
    y = Vector(np.random.randn(N,3), np.random.randn(N) < -0.5)
    z = y.perp(x)

    zero_mask = (np.random.randn(N) < -0.5) # Insert some zero-valued vectors
    x[zero_mask] = Vector.ZERO3
    z = y.perp(x)

    self.assertTrue(np.all(z.mask == (x.mask | y.mask | zero_mask)))

    # Test the unmasked items
    xx = x[~z.mask]
    zz = z[~z.mask]
    for i in range(len(zz)):
        self.assertAlmostEqual(zz[i].cross(xx[i]).norm(),
                               zz[i].norm() * xx[i].norm(), delta=1.e-14)
        self.assertAlmostEqual(zz[i].dot(xx[i]), 0., delta=1.e-14)

    # Test units
    N = 100
    x = Vector(np.random.randn(N,3), units=Units.KM)
    y = Vector(np.random.randn(N,3), units=Units.SECONDS**(-1))
    z = y.perp(x)

    self.assertEqual(z.units, Units.SECONDS**(-1))

    # Derivatives, denom = ()
    N = 100
    x = Vector(np.random.randn(N*3).reshape((N,3)))
    y = Vector(np.random.randn(N*3).reshape((N,3)))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.perp(x)

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
    z1 = y.perp(x + (EPS,0,0))
    z0 = y.perp(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.perp(x + (0,EPS,0))
    z0 = y.perp(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.perp(x + (0,0,EPS))
    z0 = y.perp(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).perp(x)
    z0 = (y - (EPS,0,0)).perp(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).perp(x)
    z0 = (y - (0,EPS,0)).perp(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).perp(x)
    z0 = (y - (0,0,EPS)).perp(x)
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

    DEL = 1.e-5
    for i in range(N):
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k], dz_df.values[i,k], delta=DEL)
        self.assertAlmostEqual(z.d_dg.values[i,k], dz_dg.values[i,k], delta=DEL)
        self.assertAlmostEqual(z.d_dh.values[i,k], dz_dh.values[i,k], delta=DEL)

    # Derivatives, denom = (2,)
    N = 100
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))

    z = y.perp(x)

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
    z1 = y.perp(x.wod + (EPS,0,0))
    z0 = y.perp(x.wod - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.perp(x.wod + (0,EPS,0))
    z0 = y.perp(x.wod - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.perp(x.wod + (0,0,EPS))
    z0 = y.perp(x.wod - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y.wod + (EPS,0,0)).perp(x.wod)
    z0 = (y.wod - (EPS,0,0)).perp(x.wod)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y.wod + (0,EPS,0)).perp(x.wod)
    z0 = (y.wod - (0,EPS,0)).perp(x.wod)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y.wod + (0,0,EPS)).perp(x.wod)
    z0 = (y.wod - (0,0,EPS)).perp(x.wod)
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

    DEL = 1.e-5
    for i in range(N):
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k,0], dz_df0.values[i,k],
                               delta=DEL)
        self.assertAlmostEqual(z.d_dg.values[i,k,0], dz_dg0.values[i,k],
                               delta=DEL)
        self.assertAlmostEqual(z.d_dh.values[i,k,0], dz_dh0.values[i,k],
                               delta=DEL)

        self.assertAlmostEqual(z.d_df.values[i,k,1], dz_df1.values[i,k],
                               delta=DEL)
        self.assertAlmostEqual(z.d_dg.values[i,k,1], dz_dg1.values[i,k],
                               delta=DEL)
        self.assertAlmostEqual(z.d_dh.values[i,k,1], dz_dh1.values[i,k],
                               delta=DEL)

    # Derivatives should be removed if necessary
    self.assertEqual(y.perp(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.perp(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.perp(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.perp(x, recursive=False), 'd_dh'))

    # Read-only status should NOT be preserved
    N = 10
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x = Vector(np.random.randn(N*3).reshape(N,3))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.perp(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().perp(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().perp(x).readonly)
    self.assertFalse(y.perp(x.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
