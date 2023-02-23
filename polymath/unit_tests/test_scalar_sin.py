################################################################################
# Scalar.sin() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_sin(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(7012)

    # Individual values
    self.assertEqual(Scalar(1.25).sin(), np.sin(1.25))
    self.assertEqual(type(Scalar(1.25).sin()), Scalar)

    self.assertEqual(Scalar(1).sin(), np.sin(1.))
    self.assertEqual(Scalar(0).sin(), 0.)

    # Multiple values
    self.assertEqual(Scalar((-1,0,1)).sin(), np.sin((-1,0,1)))
    self.assertEqual(type(Scalar((-1,0,1)).sin()), Scalar)

    # Arrays
    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    funcvals = angles.sin()
    for i in range(N):
        self.assertEqual(funcvals[i], np.sin(values[i]))

    for i in range(N-1):
        self.assertEqual(funcvals[i:i+2], np.sin(values[i:i+2]))

    # Test valid units
    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.sin, random)

    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, Scalar.sin, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.sin(), random.sin())        # units should be OK

    values = np.random.randn(10)
    random = Scalar(values, units=Units.RAD)
    self.assertEqual(random.sin(), random.sin())        # units should be OK

    angle = Scalar(3.25, units=Units.UNITLESS)
    self.assertEqual(angle.sin(), np.sin(angle.values)) # units should be OK

    # Units should be removed
    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertTrue(random.sin().units is None)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.sin()
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

    self.assertIn('t', x.sin().derivs)
    self.assertIn('vec', x.sin().derivs)
    self.assertTrue(hasattr(x.sin(), 'd_dt'))
    self.assertTrue(hasattr(x.sin(), 'd_dvec'))

    EPS = 1.e-6
    y1 = (x + EPS).sin()
    y0 = (x - EPS).sin()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.sin().d_dt
    dy_dvec = x.sin().d_dvec

    for i in range(N):
        self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], dy_dt[i], delta=1.e-5)

        for k in range(3):
            self.assertAlmostEqual(dy_dx[i] * x.d_dvec[i].values[k],
                                   dy_dvec[i].values[k], delta=1.e-5)

    # Derivatives should be removed if necessary
    self.assertEqual(x.sin(recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))
    self.assertFalse(hasattr(x.sin(recursive=False), 'd_dt'))
    self.assertFalse(hasattr(x.sin(recursive=False), 'd_dvec'))

    # Read-only status should NOT be preserved
    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    self.assertFalse(x.readonly)
    self.assertFalse(x.sin().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().sin().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
