################################################################################
# Tests for Matrix.unitary()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix3, Matrix

class Test_Matrix_unitary(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(2163)

    # Matrices 10% perturbed from unitary
    N = 100
    SCALE = 0.1
    euler = (np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi)

    a = Matrix(Matrix3.from_euler(*euler))
    a += SCALE * Matrix(np.random.randn(N,3,3))
    b = a.unitary()
    self.assertEqual(b.masked(), 0)

    # Matrices 30% perturbed from unitary
    N = 100
    SCALE = 0.3
    euler = (np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi)

    a = Matrix(Matrix3.from_euler(*euler))
    a += SCALE * Matrix(np.random.randn(N,3,3))
    b = a.unitary()
    self.assertTrue(b.masked() <= 30)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
