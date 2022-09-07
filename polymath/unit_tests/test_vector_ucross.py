################################################################################
# Vector.ucross() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

class Test_Vector_ucross(unittest.TestCase):

  # runTest
  def runTest(self):

    # Single values
    x = Vector((1.,0.,0.))
    y = Vector((0.,1.,0.))
    z = Vector((0.,0.,1.))

    self.assertEqual(x.ucross(y), z)
    self.assertEqual(y.ucross(z), x)
    self.assertEqual(z.ucross(x), y)
    self.assertFalse(x.ucross(y).mask)
    self.assertTrue(x.ucross(x).mask)

    self.assertEqual((3*x).ucross(4*y), z)
    self.assertEqual((-3*y).ucross(7*z), -x)

    # Array values
    N = 100
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))
    z = x.ucross(y)

    for i in range(N):
        self.assertAlmostEqual(x.dot(z)[i], 0., delta=1.e-12)
        self.assertAlmostEqual(y.dot(z)[i], 0., delta=1.e-12)
        self.assertAlmostEqual(z.dot(z)[i], 1., delta=1.e-12)

    # Units are stripped
    N = 10
    x = Vector(np.random.randn(N*3).reshape(N,3), units=Units.KM)
    y = Vector(np.random.randn(N*3).reshape(N,3), units=Units.SEC)
    z = x.ucross(y)
    self.assertEqual(z.units, Units.UNITLESS)

    N = 10
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))
    z = x.ucross(y)
    self.assertTrue(z.units is None)

    # Derivatives, denom = ()
    N = 6
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.ucross(x)

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
    z1 = y.ucross(x + (EPS,0,0))
    z0 = y.ucross(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.ucross(x + (0,EPS,0))
    z0 = y.ucross(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.ucross(x + (0,0,EPS))
    z0 = y.ucross(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).ucross(x)
    z0 = (y - (EPS,0,0)).ucross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).ucross(x)
    z0 = (y - (0,EPS,0)).ucross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).ucross(x)
    z0 = (y - (0,0,EPS)).ucross(x)
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

    # Read-only status should NOT be preserved
    N = 10
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x = Vector(np.random.randn(N*3).reshape(N,3))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.ucross(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().ucross(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().ucross(x).readonly)
    self.assertFalse(y.ucross(x.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
