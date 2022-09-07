################################################################################
# Scalar.cos() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_cos(unittest.TestCase):

  # runTest
  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(1.25).cos(), np.cos(1.25))
    self.assertEqual(type(Scalar(1.25).cos()), Scalar)

    self.assertEqual(Scalar(1).cos(), np.cos(1.))
    self.assertEqual(Scalar(0).cos(), 1.)

    # Multiple values
    self.assertEqual(Scalar((-1,0,1)).cos(), np.cos((-1,0,1)))
    self.assertEqual(type(Scalar((-1,0,1)).cos()), Scalar)

    # Arrays
    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    funcvals = angles.cos()
    for i in range(N):
        self.assertEqual(funcvals[i], np.cos(values[i]))

    for i in range(N-1):
        self.assertEqual(funcvals[i:i+2], np.cos(values[i:i+2]))

    # Test valid units
    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.cos, random)

    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, Scalar.cos, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.cos(), random.cos())        # units should be OK

    values = np.random.randn(10)
    random = Scalar(values, units=Units.RAD)
    self.assertEqual(random.cos(), random.cos())        # units should be OK

    angle = Scalar(3.25, units=Units.UNITLESS)
    self.assertEqual(angle.cos(), np.cos(angle.values)) # units should be OK

    # Units should be removed
    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertTrue(random.cos().units is None)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.cos()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(not np.any(y.mask[~x.mask]))

    # Derivatives
    N = 100
    x = Scalar(np.random.randn(N) * 10.)
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))

    self.assertIn('t', x.derivs)
    self.assertIn('vec', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))

    self.assertIn('t', x.cos().derivs)
    self.assertIn('vec', x.cos().derivs)
    self.assertTrue(hasattr(x.cos(), 'd_dt'))
    self.assertTrue(hasattr(x.cos(), 'd_dvec'))

    EPS = 1.e-6
    y1 = (x + EPS).cos()
    y0 = (x - EPS).cos()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.cos().d_dt
    dy_dvec = x.cos().d_dvec

    for i in range(N):
        self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], dy_dt[i], delta=1.e-5)

        for k in range(3):
            self.assertAlmostEqual(dy_dx[i] * x.d_dvec[i].values[k],
                                   dy_dvec[i].values[k], delta=1.e-5)

    # Derivatives should be removed if necessary
    self.assertEqual(x.cos(recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))
    self.assertFalse(hasattr(x.cos(recursive=False), 'd_dt'))
    self.assertFalse(hasattr(x.cos(recursive=False), 'd_dvec'))

    # Read-only status should NOT be preserved
    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    self.assertFalse(x.readonly)
    self.assertFalse(x.cos().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().cos().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
