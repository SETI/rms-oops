################################################################################
# Tests for Matrix3.to_euler() and Matrix3.from_euler()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix3

class Test_Matrix3_euler(unittest.TestCase):

  def runTest(self):

    DEL = 1.e-12

    N = 30
    euler = (np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi)

    a = Matrix3.from_euler(*euler)

    test = a * a.T
    for i in range(N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(test.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(test.values[i,j,k], int(j==k), delta=DEL)

    # Conversion to Euler angles and back always returns the same matrix
    for code in Matrix3._AXES2TUPLE.keys():
        angles = a.to_euler(axes=code)
        b = Matrix3.from_euler(*angles, axes=code)

        self.assertTrue(np.max(abs((a - b).values)) < DEL)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
