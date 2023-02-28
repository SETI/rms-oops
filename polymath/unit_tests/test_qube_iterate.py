################################################################################
# Tests for Qube item iteration
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_iterate(unittest.TestCase):

  # runTest
  def runTest(self):

    array = Scalar(np.arange(10))
    count = 0
    for a in array:
        self.assertEqual(a, count)
        self.assertTrue(isinstance(a, Scalar))
        count += 1

    array = Scalar(np.arange(10))
    count = 0
    for a in array.__iter__():
        self.assertEqual(a, count)
        self.assertTrue(isinstance(a, Scalar))
        count += 1

    array = Scalar(np.arange(10), mask=[1,1,1,1,1,0,0,0,0,0])
    count = 0
    for a in array:
        self.assertEqual(a.vals, count)
        self.assertEqual(a.mask, (count < 5))
        self.assertTrue(isinstance(a, Scalar))
        count += 1

    array = Pair(list(zip(np.arange(10), -3 * np.arange(10))))
    count = 0
    for a in array:
        self.assertEqual(a, (count, -3 * count))
        self.assertTrue(isinstance(a, Pair))
        count += 1

    count = 0
    for k,a in enumerate(array):
        self.assertEqual(a, (k, -3 * k))
        self.assertEqual(k, count)
        self.assertTrue(isinstance(a, Pair))
        count += 1

    count = 0
    for k,a in array.ndenumerate():
        self.assertEqual(a, (k[0], -3 * k[0]))
        self.assertEqual(a, array[k])
        self.assertEqual(k[0], count)
        self.assertTrue(isinstance(a, Pair))
        count += 1

    array = Scalar(np.arange(10).reshape(5,2))
    for k,a in enumerate(array):
        self.assertEqual(a, (2*k, 2*k+1))
        self.assertEqual(a, array[k])

    for k,a in array.ndenumerate():
        self.assertEqual(a, array[k])

    # shape ()
    array = Scalar(7)
    count = 0
    for a in array:
        self.assertEqual(a, array)
        count += 1
    self.assertEqual(count, 1)

    count = 0
    for k,a in array.ndenumerate():
        self.assertEqual(k[0], 0)
        self.assertEqual(a, array)
        count += 1
    self.assertEqual(count, 1)

############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
