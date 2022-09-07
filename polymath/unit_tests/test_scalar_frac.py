################################################################################
# Scalar.frac() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_frac(unittest.TestCase):

  # runTest
  def runTest(self):

    # Individual values
    self.assertEqual(Scalar( 1.25).frac(), 0.25)
    self.assertEqual(Scalar(-1.25).frac(), 0.75)
    self.assertEqual(Scalar( 1).frac(), 0.)
    self.assertEqual(Scalar(-1).frac(), 0.)

    # Multiple values
    self.assertEqual(Scalar((1.25, -1.25)).frac(), (0.25, 0.75))
    self.assertTrue(Scalar((1.25, -1.25)).frac().is_float())

    self.assertEqual(Scalar((1, -1)).frac(), (0.,0.))
    self.assertTrue(Scalar((1.2, -1.2)).frac().is_float())

    # Arrays
    N = 1000
    values = np.random.randn(N) * 10.
    random = Scalar(values)
    frandom = random.frac()
    for i in range(N):
        self.assertEqual(frandom[i], values[i] % 1.)

    for i in range(N-1):
        self.assertEqual(random[i:i+2].frac(), values[i:i+2] % 1.)

    # Units should be disallowed
    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.frac, random)

    random = Scalar(values, units=Units.DEG)
    self.assertRaises(ValueError, Scalar.frac, random)

    random = Scalar(3.25, units=Units.UNITLESS)
    self.assertEqual(random.frac(), 0.25)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.frac()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(not np.any(y.mask[~x.mask]))

    # Derivatives should be preserved
    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)),
                                       drank=1))
    self.assertIn('t', random.derivs)
    self.assertIn('vec', random.derivs)
    self.assertTrue(hasattr(random, 'd_dt'))
    self.assertTrue(hasattr(random, 'd_dvec'))

    self.assertEqual(random.frac().derivs, random.derivs)
    self.assertIn('t', random.frac().derivs)
    self.assertIn('vec', random.frac().derivs)
    self.assertTrue(hasattr(random.frac(), 'd_dt'))
    self.assertTrue(hasattr(random.frac(), 'd_dvec'))

    N = 10
    random = Scalar(np.arange(10))
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape(N,3),
                                       drank=1))
    self.assertIn('t', random.derivs)
    self.assertIn('vec', random.derivs)
    self.assertTrue(hasattr(random, 'd_dt'))
    self.assertTrue(hasattr(random, 'd_dvec'))

    self.assertEqual(random.frac().derivs, random.derivs)
    self.assertIn('t', random.frac().derivs, 't')
    self.assertIn('vec', random.frac().derivs, 'vec')
    self.assertTrue(hasattr(random.frac(), 'd_dt'))
    self.assertTrue(hasattr(random.frac(), 'd_dvec'))

    # Read-only status should NOT be preserved
    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    self.assertFalse(random.readonly)
    self.assertFalse(random.frac().readonly)
    self.assertTrue(random.as_readonly().readonly)
    self.assertFalse(random.as_readonly().frac().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
