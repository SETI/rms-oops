################################################################################
# Scalar.min() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_min(unittest.TestCase):

  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(0.3).min(), 0.3)
    self.assertEqual(type(Scalar(0.3).min()), float)

    self.assertEqual(Scalar(4).min(), 4)
    self.assertEqual(type(Scalar(4).min()), int)

    self.assertTrue(Scalar(4, mask=True).min().mask)
    self.assertEqual(type(Scalar(4, mask=True).min()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).min() == 1)
    self.assertEqual(type(Scalar((1,2,3)).min()), int)

    self.assertTrue(Scalar((1.,2.,3.)).min() == 1.)
    self.assertEqual(type(Scalar((1.,2,3)).min()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.min(), np.min(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.min().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.min().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.min()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    minval = np.inf
    for i in range(N):
        if (not x.mask[i]) and (x.values[i] < minval):
            minval = x.values[i]

    self.assertEqual(minval, x.min())

    # If we mask the minimum value(s), the minimum should increase
    x = x.mask_where_eq(minval)
    self.assertTrue(x.min() > minval)

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.min().mask)
    self.assertTrue(type(masked.min()), Scalar)

    # Denominators
    a = Scalar([1.,2.], drank=1)
    self.assertRaises(ValueError, a.min)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
