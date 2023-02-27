################################################################################
# Scalar.sign() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_sign(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(5251)

    # Individual values
    self.assertEqual(Scalar(1.25).sign(), 1.)
    self.assertEqual(type(Scalar(1.25).sign()), Scalar)

    self.assertEqual(Scalar(1).sign(), np.sign(1.))
    self.assertEqual(Scalar(0).sign(), 0.)

    # Multiple values
    self.assertEqual(Scalar((-1,0,1)).sign(), np.sign((-1,0,1)))
    self.assertEqual(type(Scalar((-1,0,1)).sign()), Scalar)

    # Arrays
    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.sign()
    for i in range(N):
        self.assertEqual(y[i], np.sign(x.values[i]))

    for i in range(N-1):
        self.assertEqual(y[i:i+2], np.sign(x.values[i:i+2]))

    # Test valid units
    values = np.random.randn(10)
    x = Scalar(values, units=Units.KM)
    self.assertEqual(x.sign(), np.sign(values))

    values = np.random.randn(10)
    x = Scalar(values, units=Units.SECONDS)
    self.assertEqual(x.sign(), np.sign(values))

    values = np.random.randn(10)
    x = Scalar(values, units=Units.DEG)
    self.assertEqual(x.sign(), np.sign(values))

    values = np.random.randn(10)
    x = Scalar(values, units=Units.UNITLESS)
    self.assertEqual(x.sign(), np.sign(values))

    # Units should be removed
    values = np.random.randn(10)
    x = Scalar(values, units=Units.CM)
    self.assertTrue(x.sign().units is None)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.sign()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(not np.any(y.mask[~x.mask]))

    # Derivatives are removed
    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))

    self.assertIn('t', x.derivs)
    self.assertIn('vec', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))

    self.assertNotIn('t', x.sign().derivs)
    self.assertNotIn('vec', x.sign().derivs)
    self.assertFalse(hasattr(x.sign(), 'd_dt'))
    self.assertFalse(hasattr(x.sign(), 'd_dvec'))

    # Read-only status should NOT be preserved
    N = 10
    x = Scalar(np.random.randn(N))
    self.assertFalse(x.readonly)
    self.assertFalse(x.sign().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().sign().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
