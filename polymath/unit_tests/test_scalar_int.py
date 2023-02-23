################################################################################
# Scalar.int() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_int(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(4353)

    # Individual values
    self.assertEqual(Scalar( 1.2).int(),  1)
    self.assertEqual(Scalar(-1.2).int(), -2)
    self.assertEqual(Scalar( 1).int(),  1)
    self.assertEqual(Scalar(-1).int(), -1)

    self.assertEqual(Scalar(1.2,True).int(), Scalar(0.).masked_single())
    self.assertEqual(Scalar(1,  True).int(), Scalar(0.).masked_single())

    # Multiple values
    self.assertEqual(Scalar((1.2, -1.2)).int(), (1,-2))
    self.assertFalse(Scalar((1.2, -1.2)).int().is_float())

    self.assertEqual(Scalar((1, -1)).int(), (1,-1))
    self.assertFalse(Scalar((1.2, -1.2)).int().is_float())

    # Arrays
    N = 1000
    values = np.random.randn(N) * 10.
    random = Scalar(values)
    irandom = random.int()
    for i in range(N):
        self.assertEqual(irandom[i], int(np.floor(values[i])))

    for i in range(N-1):
        self.assertEqual(random[i:i+2].int(), np.floor(values[i:i+2]))

    # Units should be disallowed
    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.int, random)

    random = Scalar(values, units=Units.DEG)
    self.assertRaises(ValueError, Scalar.int, random)

    random = Scalar(3.14, units=Units.UNITLESS)
    self.assertEqual(random.int(), 3)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.int()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(not np.any(y.mask[~x.mask]))

    # Derivatives should be stripped
    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape(N,3),
                                       drank=1))
    self.assertIn('t', random.derivs)
    self.assertIn('vec', random.derivs)
    self.assertTrue(hasattr(random, 'd_dt'))
    self.assertTrue(hasattr(random, 'd_dvec'))

    self.assertEqual(random.int().derivs, {})
    self.assertNotIn('t', random.int().derivs)
    self.assertNotIn('vec', random.int().derivs)
    self.assertFalse(hasattr(random.int(), 'd_dt'))
    self.assertFalse(hasattr(random.int(), 'd_dvec'))

    N = 10
    random = Scalar(np.arange(10))
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)),
                                       drank=1))
    self.assertIn('t', random.derivs)
    self.assertIn('vec', random.derivs)
    self.assertTrue(hasattr(random, 'd_dt'))
    self.assertTrue(hasattr(random, 'd_dvec'))

    self.assertEqual(random.int().derivs, {})
    self.assertNotIn('t', random.int().derivs, 't')
    self.assertNotIn('vec', random.int().derivs, 'vec')
    self.assertFalse(hasattr(random.int(), 'd_dt'))
    self.assertFalse(hasattr(random.int(), 'd_dvec'))

    # Read-only status should NOT be preserved
    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    self.assertFalse(random.readonly)
    self.assertFalse(random.int().readonly)
    self.assertTrue(random.as_readonly().readonly)
    self.assertFalse(random.as_readonly().int().readonly)

    # But int objects are returned as is
    a = Scalar(np.arange(10)).as_readonly()
    self.assertTrue(a.readonly)
    self.assertTrue(a.int().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
