################################################################################
# Scalar.tan() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_tan(unittest.TestCase):

  # runTest
  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(1.25).tan(), np.tan(1.25))
    self.assertEqual(type(Scalar(1.25).tan()), Scalar)

    self.assertEqual(Scalar(1).tan(), np.tan(1.))
    self.assertEqual(Scalar(0).tan(), 0.)

    # Multiple values
    self.assertEqual(Scalar((-1,0,1)).tan(), np.tan((-1,0,1)))
    self.assertEqual(type(Scalar((-1,0,1)).tan()), Scalar)

    # Arrays
    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    for i in range(N):
        self.assertEqual(angles.tan()[i], np.tan(values[i]))

    for i in range(N-1):
        self.assertEqual(angles.tan()[i:i+2], np.tan(values[i:i+2]))

    # Test valid units
    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.tan, random)

    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, Scalar.tan, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.tan(), random.tan())        # units should be OK

    values = np.random.randn(10)
    random = Scalar(values, units=Units.RAD)
    self.assertEqual(random.tan(), random.tan())        # units should be OK

    angle = Scalar(3.25, units=Units.UNITLESS)
    self.assertEqual(angle.tan(), np.tan(angle.values)) # units should be OK

    # Units should be removed
    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertTrue(random.tan().units is None)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.tan()
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

    self.assertIn('t', x.tan().derivs)
    self.assertIn('vec', x.tan().derivs)
    self.assertTrue(hasattr(x.tan(), 'd_dt'))
    self.assertTrue(hasattr(x.tan(), 'd_dvec'))

    EPS = 1.e-6
    y1 = (x + EPS).tan()
    y0 = (x - EPS).tan()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.tan().d_dt
    dy_dvec = x.tan().d_dvec

    DEL = 5.e-5
    for i in range(N):
        scale = dy_dt[i]
        self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], dy_dt[i],
                               delta = DEL * abs(dy_dt[i]))

        for k in range(3):
            self.assertAlmostEqual(dy_dx[i] * x.d_dvec[i].values[k],
                                   dy_dvec[i].values[k],
                                   delta = DEL * abs(dy_dvec[i].values[k]))

    # Read-only status should NOT be preserved
    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    self.assertFalse(x.readonly)
    self.assertFalse(x.tan().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().tan().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
