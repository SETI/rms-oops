################################################################################
# Scalar.maximum() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar

class Test_Scalar_maximum(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(4187)

    self.assertRaises(ValueError, Scalar.maximum)

    a = Scalar(np.random.randn(10,1))
    self.assertEqual(Scalar.maximum(a), a)
    self.assertEqual(Scalar.maximum(a,-100), a)
    self.assertEqual(Scalar.maximum(a,-100,Scalar.MASKED), a)

    b = Scalar(np.random.randn(4,1,10))
    self.assertEqual(Scalar.maximum(a,b).shape, (4,10,10))

    ab = Scalar.maximum(a,b,-100,Scalar.MASKED)
    ab2 = Scalar(np.maximum(a.values,b.values))
    self.assertEqual(ab, ab2)

    a = Scalar(np.random.randn(10,1), np.random.randn(10,1) < -0.5)
    b = Scalar(np.random.randn(4,1,10), np.random.randn(4,1,10) < -0.5)
    ab = Scalar.maximum(a,b)

    for i in range(4):
      for j in range(10):
        for k in range(10):
            if a.mask[j,0] and b.mask[i,0,k]:
                self.assertTrue(ab[i,j,k].mask)
            elif a.mask[j,0]:
                self.assertEqual(ab[i,j,k].vals, b[i,0,k].vals)
                self.assertFalse(ab[i,j,k].mask)
            elif b.mask[i,0,k]:
                self.assertEqual(ab[i,j,k].vals, a[j,0].vals)
                self.assertFalse(ab[i,j,k].mask)
            else:
                self.assertEqual(ab[i,j,k], max(a[j,0],b[i,0,k]))
                self.assertFalse(ab[i,j,k].mask)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
