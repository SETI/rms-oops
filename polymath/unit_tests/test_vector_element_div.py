################################################################################
# Vector.element_div()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Vector, Scalar, Units

class Test_Vector_element_div(unittest.TestCase):

  def runTest(self):

    # Single values
    self.assertEqual(Vector((2,21,0)).element_div((1,3,1)), (2,7,0))
    self.assertEqual(Vector((20,30,40)).element_div((10,10,-20)), (2,3,-2))
    self.assertTrue(Vector((2,3,0),True).element_div((10,10,-20)).mask)
    self.assertTrue(Vector((2,3,0),False).element_div((10,10,0)).mask)

    # Arrays and masks
    N = 100
    x = Vector(np.random.randn(N,5))
    y = Vector(np.random.randn(N,5))
    z = y.element_div(x)

    DEL = 3.e-12
    for i in range(N):
        for k in range(5):
            self.assertAlmostEqual(z[i], y.values[i]/x.values[i], delta=DEL)

    N = 100
    x = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    y = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    z = y.element_div(x)

    self.assertTrue(np.all(z.mask == (x.mask | y.mask)))

    # Compare the unmasked values
    zz = z[~z.mask]
    xx = x[~z.mask]
    yy = y[~z.mask]
    for i in range(len(zz)):
      for k in range(4):
        self.assertAlmostEqual(zz[i], yy.values[i]/xx.values[i], delta=DEL)

    N = 100
    x = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    y = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)

    zero_mask = (np.random.randn(N,4) < -1.)
    x.values[zero_mask] = 0.

    zero_mask = np.any(zero_mask, axis=-1)

    z = y.element_div(x)

    self.assertTrue(np.all(z.mask[x.mask]))
    self.assertTrue(np.all(z.mask[y.mask]))
    self.assertTrue(np.all(z.mask[zero_mask]))
    self.assertTrue(np.all(z.mask == (x.mask | y.mask | zero_mask)))

    for i in range(N):
      for k in range(4):
        if not z[i].mask:
             self.assertAlmostEqual(z[i], y.values[i]/x.values[i], delta=DEL)

    # Test units
    N = 100
    x = Vector(np.random.randn(N,3), units=Units.S)
    y = Vector(np.random.randn(N,3), units=Units.KM)
    z = y.element_div(x)

    self.assertEquals(z.units, Units.KM/Units.SECONDS)

    # Derivatives, denom = ()
    N = 100
    x = Vector(np.random.randn(N*3).reshape((N,3)))
    y = Vector(np.random.randn(N*3).reshape((N,3)))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.element_div(x)

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

    # Construct the numerical derivative dz/dx
    EPS = 1.e-6
    z1 = y.element_div(x + (EPS,0,0))
    z0 = y.element_div(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.element_div(x + (0,EPS,0))
    z0 = y.element_div(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.element_div(x + (0,0,EPS))
    z0 = y.element_div(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    new_values = np.empty((N,3,3))
    new_values[...,0] = dz_dx0.values
    new_values[...,1] = dz_dx1.values
    new_values[...,2] = dz_dx2.values

    dz_dx = Vector(new_values, drank=1)

    # Construct the numerical derivative dz/dy
    z1 = (y + (EPS,0,0)).element_div(x)
    z0 = (y - (EPS,0,0)).element_div(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).element_div(x)
    z0 = (y - (0,EPS,0)).element_div(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).element_div(x)
    z0 = (y - (0,0,EPS)).element_div(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS

    new_values = np.empty((N,3,3))
    new_values[...,0] = dz_dy0.values
    new_values[...,1] = dz_dy1.values
    new_values[...,2] = dz_dy2.values

    dz_dy = Vector(new_values, drank=1)

    dz_df = dz_dx.chain(x.d_df)
    dz_dg = dz_dy.chain(y.d_dg)
    dz_dh = dz_dx.chain(x.d_dh) + dz_dy.chain(y.d_dh)

    DEL = 1.e-4
    for i in range(N):
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k], dz_df.values[i,k],
                               delta = max(1., abs(dz_df.values[i,k])) * DEL)
        self.assertAlmostEqual(z.d_dg.values[i,k], dz_dg.values[i,k],
                               delta = max(1., abs(dz_dg.values[i,k])) * DEL)
        self.assertAlmostEqual(z.d_dh.values[i,k], dz_dh.values[i,k],
                               delta = max(1., abs(dz_dh.values[i,k])) * DEL)

    # Derivatives, denom = (2,)
    N = 300
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))

    z = y.element_div(x)

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
    z1 = y.element_div(x + (EPS,0,0))
    z0 = y.element_div(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.element_div(x + (0,EPS,0))
    z0 = y.element_div(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.element_div(x + (0,0,EPS))
    z0 = y.element_div(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).element_div(x)
    z0 = (y - (EPS,0,0)).element_div(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).element_div(x)
    z0 = (y - (0,EPS,0)).element_div(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).element_div(x)
    z0 = (y - (0,0,EPS)).element_div(x)
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

    DEL = 1.e-3
    for i in range(N):
      for k in range(3):
        self.assertAlmostEqual(z.d_df.values[i,k,0], dz_df0.values[i,k],
                               delta = max(1., abs(dz_df0.values[i,k])) * DEL)
        self.assertAlmostEqual(z.d_dg.values[i,k,0], dz_dg0.values[i,k],
                               delta = max(1., abs(dz_dg0.values[i,k])) * DEL)
        self.assertAlmostEqual(z.d_dh.values[i,k,0], dz_dh0.values[i,k],
                               delta = max(1., abs(dz_dh0.values[i,k])) * DEL)

        self.assertAlmostEqual(z.d_df.values[i,k,1], dz_df1.values[i,k],
                               delta = max(1., abs(dz_df1.values[i,k])) * DEL)
        self.assertAlmostEqual(z.d_dg.values[i,k,1], dz_dg1.values[i,k],
                               delta = max(1., abs(dz_dg1.values[i,k])) * DEL)
        self.assertAlmostEqual(z.d_dh.values[i,k,1], dz_dh1.values[i,k],
                               delta = max(1., abs(dz_dh1.values[i,k])) * DEL)

    # Derivatives should be removed if necessary
    self.assertEqual(y.element_div(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.element_div(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.element_div(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.element_div(x, recursive=False), 'd_dh'))

    # Read-only status should NOT be preserved
    N = 10
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x = Vector(np.random.randn(N*3).reshape(N,3))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.element_div(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().element_div(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().element_div(x).readonly)
    self.assertFalse(y.element_div(x.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
