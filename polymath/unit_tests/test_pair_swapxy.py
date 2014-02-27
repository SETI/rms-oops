################################################################################
# Pair.swapxy() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Pair, Scalar, Units

class Test_Pair_swapxy(unittest.TestCase):

  def runTest(self):

    # Single values
    a = Pair((1,2))
    b = a.swapxy()
    self.assertEqual(b, (2,1))
    self.assertTrue(a.mask is b.mask)

    # Arrays & denoms
    N = 10
    a = Pair(np.arange(N*6).reshape(N,2,3), drank=1)
    b = a.swapxy()

    aparts = a.to_scalars()
    bparts = b.to_scalars()

    self.assertEqual(aparts[0], bparts[1])
    self.assertEqual(aparts[1], bparts[0])

    # Masks
    a = Pair(np.random.randn(N,2,3), drank=1,
             mask = (np.random.randn(N) < -0.4))
    b = a.swapxy()
    self.assertTrue(np.all(a.mask == b.mask))

    # Units
    N = 10
    a = Pair(np.arange(N*6).reshape(N,2,3), drank=1, units=Units.DEG)
    b = a.swapxy()
    self.assertEqual(b.units, a.units)

    # Derivatives, denom = ()
    N = 100
    a = Pair(np.random.randn(N,2))

    a.insert_deriv('t', Pair(np.random.randn(N,2)))
    a.insert_deriv('v', Pair(np.random.randn(N,2,3), drank=1,
                             mask = (np.random.randn(N) < -0.4)))

    self.assertIn('t', a.derivs)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertIn('v', a.derivs)
    self.assertTrue(hasattr(a, 'd_dv'))

    b = a.swapxy(recursive=False)
    self.assertNotIn('t', b.derivs)
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertNotIn('v', b.derivs)
    self.assertFalse(hasattr(b, 'd_dv'))

    b = a.swapxy()
    self.assertIn('t', b.derivs)
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertIn('v', b.derivs)
    self.assertTrue(hasattr(b, 'd_dv'))

    EPS = 1.e-6
    b1 = (a + (EPS,0)).swapxy()
    b0 = (a - (EPS,0)).swapxy()
    db_da0 = 0.5 * (b1 - b0) / EPS

    b1 = (a + (0,EPS)).swapxy()
    b0 = (a - (0,EPS)).swapxy()
    db_da1 = 0.5 * (b1 - b0) / EPS

    db_dt = (db_da0 * a.d_dt.values[:,0] +
             db_da1 * a.d_dt.values[:,1])

    db_dv0 = (db_da0 * a.d_dv.values[:,0,0] +
              db_da1 * a.d_dv.values[:,1,0])

    db_dv1 = (db_da0 * a.d_dv.values[:,0,1] +
              db_da1 * a.d_dv.values[:,1,1])

    db_dv2 = (db_da0 * a.d_dv.values[:,0,2] +
              db_da1 * a.d_dv.values[:,1,2])

    DEL = 1.e-5
    for i in range(N):
      for k in range(2):
        self.assertAlmostEqual(b.d_dt.values[i,k],
                               db_dt.values[i,k], delta=DEL)
        self.assertAlmostEqual(b.d_dv.values[i,k,0],
                               db_dv0.values[i,k], delta=DEL)
        self.assertAlmostEqual(b.d_dv.values[i,k,1],
                               db_dv1.values[i,k], delta=DEL)
        self.assertAlmostEqual(b.d_dv.values[i,k,2],
                               db_dv2.values[i,k], delta=DEL)

    da_dt_parts = a.d_dt.to_scalars()
    db_dt_parts = b.d_dt.to_scalars()
    self.assertEqual(da_dt_parts[0], db_dt_parts[1])
    self.assertEqual(da_dt_parts[1], db_dt_parts[0])

    da_dv_parts = a.d_dv.to_scalars()
    db_dv_parts = b.d_dv.to_scalars()
    self.assertEqual(da_dv_parts[0], db_dv_parts[1])
    self.assertEqual(da_dv_parts[1], db_dv_parts[0])

    # Read-only status should be preserved
    N = 10
    a = Pair(np.random.randn(N,2))
    b = Pair(np.random.randn(N,2))

    self.assertFalse(a.readonly)
    self.assertFalse(a.swapxy().readonly)
    self.assertTrue(a.as_readonly().swapxy().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
