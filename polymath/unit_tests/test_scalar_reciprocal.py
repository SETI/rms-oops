################################################################################
# Scalar.reciprocal() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_reciprocal(unittest.TestCase):

  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(1.).reciprocal(), 1.)
    self.assertEqual(type(Scalar(1.).reciprocal()), Scalar)

    self.assertEqual(Scalar(0.5).reciprocal(), 2.)

    self.assertTrue(Scalar(0.).reciprocal().mask)
    self.assertEqual(type(Scalar(0.).reciprocal()), Scalar)

    # Multiple values
    self.assertEqual(Scalar((-1,1)).reciprocal(), (-1,1))

    # Arrays
    N = 1000
    x = Scalar(np.random.randn(N))
    self.assertTrue(x.reciprocal().mask is False)
    self.assertEqual(np.shape(x.reciprocal().mask), ())
    self.assertEqual(x.reciprocal(), 1./x.values)

    x = Scalar(np.random.randn(N))
    x.values[np.random.randn(N) < -0.5] = 0.
    self.assertEqual(np.shape(x.reciprocal().mask), (N,))
    self.assertTrue(np.sum(x.reciprocal().mask) > 0)

    # Test valid units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.reciprocal().units, Units.KM**(-1))

    values = np.random.randn(10)
    random = Scalar(values, units=Units.SECONDS)
    self.assertEqual(random.reciprocal().units, Units.SECONDS**(-1))

    values = np.random.randn(10)
    random = Scalar(values, units=Units.UNITLESS)
    self.assertEqual(random.reciprocal().units, Units.UNITLESS)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertTrue(random.reciprocal().units is None)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    zero_mask = np.random.randn(N) < -0.5
    x.values[zero_mask] = 0.
    self.assertTrue(np.sum(x.mask) > 0)
    self.assertTrue(np.sum(zero_mask) > 0)
    
    y = x.reciprocal()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(np.all(y.mask[zero_mask]))
    self.assertTrue(not np.any(y.mask[~zero_mask & ~x.mask]))

    # Derivatives
    N = 1000
    x = Scalar(np.random.randn(N))
    zero_mask = np.random.randn(N) < -0.5
    x.values[zero_mask] = 0.

    x.insert_deriv('t', Scalar(np.random.randn(N)))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))

    self.assertIn('t', x.derivs)
    self.assertIn('vec', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))

    self.assertIn('t', x.reciprocal().derivs)
    self.assertIn('vec', x.reciprocal().derivs)
    self.assertTrue(hasattr(x.reciprocal(), 'd_dt'))
    self.assertTrue(hasattr(x.reciprocal(), 'd_dvec'))

    EPS = 1.e-6
    y1 = (x + EPS).reciprocal()
    y0 = (x - EPS).reciprocal()
    y = x.reciprocal()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.reciprocal().d_dt
    dy_dvec = x.reciprocal().d_dvec

    DEL = 3.e-6
    for i in range(N):
        if not y.mask[i]:
            deriv = dy_dt[i]
            if abs(deriv) < 1.e5:
                self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], deriv,
                                       delta = DEL * abs(deriv))

            for k in range(3):
                deriv = dy_dvec[i].values[k]
                if abs(deriv) < 1.e5:
                    self.assertAlmostEqual(dy_dx[i] * x.d_dvec[i].values[k],
                                           deriv, delta = DEL * abs(deriv))

    # Derivatives should be removed if necessary
    self.assertEqual(x.reciprocal(recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))
    self.assertFalse(hasattr(x.reciprocal(recursive=False), 'd_dt'))
    self.assertFalse(hasattr(x.reciprocal(recursive=False), 'd_dvec'))

    # Read-only status should be preserved
    N = 10
    x = Scalar(np.random.randn(N))
    self.assertFalse(x.readonly)
    self.assertFalse(x.reciprocal().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(x.as_readonly().reciprocal().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
