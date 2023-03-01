################################################################################
# Scalar.arctan() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_arctan(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(6021)

    # Individual values
    self.assertEqual(Scalar(-0.3).arctan(), np.arctan(-0.3))
    self.assertEqual(type(Scalar(-0.3).arctan()), Scalar)

    self.assertEqual(Scalar(0.).arctan(), np.arctan(0.))
    self.assertEqual(Scalar(0).arctan(), 0.)

    # Multiple values
    self.assertEqual(Scalar((-0.1,0.,0.1)).arctan(), np.arctan((-0.1,0.,0.1)))
    self.assertEqual(type(Scalar((-0.1,0.,0.1)).arctan()), Scalar)

    # Arrays
    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.arctan()
    for i in range(N):
        self.assertEqual(y[i], np.arctan(x.values[i]))

    for i in range(N-1):
        if np.all(np.abs(x.values[i:i+2]) <= 1):
            self.assertEqual(y[i:i+2], np.arctan(x.values[i:i+2]))

    # Test valid units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.arctan, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, Scalar.arctan, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertRaises(ValueError, Scalar.arctan, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.RAD)
    self.assertRaises(ValueError, Scalar.arctan, random)

    x = Scalar(3.25, units=Units.UNITLESS)
    self.assertFalse(x.arctan().mask)

    # Units should be removed
    values = np.random.randn(10)
    random = Scalar(values, units=Units.UNITLESS)
    self.assertTrue(random.arctan().units is None)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.arctan()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(not np.any(y.mask[~x.mask]))

    # Derivatives
    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))

    self.assertIn('t', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))

    self.assertIn('t', x.arctan().derivs)
    self.assertTrue(hasattr(x.arctan(), 'd_dt'))

    EPS = 1.e-6
    y1 = (x + EPS).arctan()
    y0 = (x - EPS).arctan()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.arctan().d_dt

    for i in range(N):
        self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], dy_dt[i], delta=EPS)

    # Derivatives should be removed if necessary
    self.assertEqual(x.arctan(recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertFalse(hasattr(x.arctan(recursive=False), 'd_dt'))

    # Read-only status should NOT be preserved
    N = 10
    x = Scalar(np.random.randn(N))
    self.assertFalse(x.readonly)
    self.assertFalse(x.arctan().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().arctan().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
