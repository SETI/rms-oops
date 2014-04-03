################################################################################
# Scalar.mean() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_mean(unittest.TestCase):

  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(0.3).mean(), 0.3)
    self.assertEqual(type(Scalar(0.3).mean()), float)

    self.assertEqual(Scalar(4).mean(), 4)
    self.assertEqual(type(Scalar(4).mean()), int)

    self.assertTrue(Scalar(4, mask=True).mean().mask)
    self.assertEqual(type(Scalar(4, mask=True).mean()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).mean() == 2)
    self.assertEqual(type(Scalar((1,2,3)).mean()), int)

    self.assertTrue(Scalar((1,2,3,4)).mean() == 2.5)
    self.assertEqual(type(Scalar((1,2,3,4)).mean()), float)

    self.assertTrue(Scalar((1.,2.,3.)).mean() == 2.)
    self.assertEqual(type(Scalar((1.,2,3)).mean()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.mean(), np.mean(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.mean().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.mean().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.mean()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    meanval = 0.
    count = 0
    for i in range(N):
        if not x.mask[i]:
            count += 1
            meanval += x.values[i]

    meanval /= count
    self.assertEqual(meanval, x.mean())

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.mean().mask)
    self.assertTrue(type(masked.mean()), Scalar)

    # Denominators and derivatives
    a = Scalar([1,2,3,4])
    b = Scalar([[1,2],[3,4],[5,6],[7,8]], drank=1)
    self.assertEqual(a.mean(), 2.5)
    self.assertEqual(b.mean(), (4,5))
    self.assertTrue(b.is_int())
    self.assertTrue(b.mean().is_int())

    a.insert_deriv('t', b)
    mean = a.mean()
    self.assertEqual(mean, 2.5)
    self.assertEqual(type(mean), float)

    mean = a.mean(True)
    self.assertEqual(mean, 2.5)
    self.assertEqual(type(mean), Scalar)
    self.assertEqual(mean.d_dt, (4,5))
    self.assertTrue(mean.d_dt.is_int())

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
