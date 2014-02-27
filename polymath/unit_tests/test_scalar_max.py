################################################################################
# Scalar.max() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_max(unittest.TestCase):

  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(0.3).max(), 0.3)
    self.assertEqual(type(Scalar(0.3).max()), float)

    self.assertEqual(Scalar(4).max(), 4)
    self.assertEqual(type(Scalar(4).max()), int)

    self.assertTrue(Scalar(4, mask=True).max().mask)
    self.assertEqual(type(Scalar(4, mask=True).max()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).max() == 3)
    self.assertEqual(type(Scalar((1,2,3)).max()), int)

    self.assertTrue(Scalar((1.,2.,3.)).max() == 3.)
    self.assertEqual(type(Scalar((1.,2,3)).max()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.max(), np.max(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.max().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.max().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.max()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    maxval = -np.inf
    for i in range(N):
        if (not x.mask[i]) and (x.values[i] > maxval):
            maxval = x.values[i]

    self.assertEqual(maxval, x.max())

    # If we mask the maximum value(s), the maximum should decrease
    x.mask[x.values == maxval] = -np.inf
    self.assertTrue(x.max() < maxval)

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.max().mask)
    self.assertTrue(type(masked.max()), Scalar)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
