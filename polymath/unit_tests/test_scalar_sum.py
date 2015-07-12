################################################################################
# Scalar.mean() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_sum(unittest.TestCase):

  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(0.3).sum(), 0.3)
    self.assertEqual(type(Scalar(0.3).sum()), float)

    self.assertEqual(Scalar(4).sum(), 4)
    self.assertEqual(type(Scalar(4).sum()), int)

    self.assertTrue(Scalar(4, mask=True).sum().mask)
    self.assertEqual(type(Scalar(4, mask=True).sum()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).sum() == 6)
    self.assertEqual(type(Scalar((1,2,3)).sum()), int)

    self.assertTrue(Scalar((1.,2.,3.)).sum() == 6.)
    self.assertEqual(type(Scalar((1.,2,3)).sum()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.sum(), np.sum(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.sum().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.sum().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.sum()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    sumval = 0.
    for i in range(N):
        if not x.mask[i]:
            sumval += x.values[i]

    self.assertTrue(abs((sumval - x.sum()) / sumval) < 4.e-15)

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.sum().mask)
    self.assertTrue(type(masked.sum()), Scalar)

    # Denominators
    a = Scalar([1.,2.], drank=1)
    self.assertRaises(ValueError, a.sum)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
