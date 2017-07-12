################################################################################
# Matrix.is_diagonal() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Boolean

class Test_Matrix_is_diagonal(unittest.TestCase):

  def runTest(self):

    N = 4
    mats = np.random.randn(N,5,5)
    self.assertEqual(Matrix(mats).is_diagonal(), False)

    mats = np.zeros((N,4,4))
    self.assertEqual(Matrix(mats).is_diagonal(), True)

    # must be square
    mats = np.empty((N,2,3))
    self.assertRaises(ValueError, Matrix(mats).is_diagonal)

    # can't have a denominator
    mats = np.empty((N,3,3,2))
    self.assertRaises(ValueError, Matrix(mats, drank=1).is_diagonal)

    # delta = 0
    mats = np.zeros((N,3,3))
    for i in range(N):
        for j in range(3):
            mats[i,j,j] = np.random.randn()

    self.assertEqual(Matrix(mats).is_diagonal(), True)

    mats[0,0,1] = 1.e-14
    self.assertEqual(Matrix(mats).is_diagonal(), [False] + (N-1)*[True])

    # delta = 1.e-13
    self.assertEqual(Matrix(mats).is_diagonal(delta=1.e-13), True)

    # all masked
    self.assertEqual(Matrix(np.random.randn(N,5,5),True).is_diagonal(), True)
    self.assertEqual(Matrix(np.random.randn(5,5),True).is_diagonal(), True)

    # masked elements
    self.assertEqual(Matrix(mats).is_diagonal(), [False] + (N-1)*[True])

    mask = [True] + (N-1) * [False]
    self.assertEqual(Matrix(mats,mask).is_diagonal(), True)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
