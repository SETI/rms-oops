################################################################################
# Scalar.as_index() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Vector, Boolean, Units

class Test_Scalar_as_index(unittest.TestCase):

  def runTest(self):

    a = Scalar(np.arange(12).reshape(3,4))
    self.assertTrue(np.all(a.as_index() == a.values))

    mask = a.values % 2 == 0
    a = Scalar(np.arange(12).reshape(3,4), mask)
    self.assertTrue(np.all(a.as_index() == np.arange(1,12,2)))

    test = a.as_index(masked=-7)
    self.assertEqual(test.shape, (3,4))
    for i in range(3):
        for j in range(4):
            if mask[i,j]:
                test[i,j] == -7
            else:
                test[i,j] == a.values[i,j]

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
