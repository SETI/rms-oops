################################################################################
# Tests for Quaternion.to_euler() and Quaternion.from_euler()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Quaternion, Matrix3

class Test_Quaternion_euler(unittest.TestCase):

  # runTest
  def runTest(self):

    # Quaternion to Euler and back, one Quaternion
    for code in Quaternion._AXES2TUPLE.keys():
        a = Quaternion(np.random.rand(4)).unit()
        euler = a.to_euler(code)
        b = Quaternion.from_euler(*euler, axes=code)

    DEL = 1.e-14
    for j in range(4):
        self.assertAlmostEqual(a.values[j], b.values[j], delta=DEL)

    # Quaternion to Euler and back, N Quaternions
    N = 100
    for code in Quaternion._AXES2TUPLE.keys():
        a = Quaternion(np.random.rand(N,4)).unit()
        euler = a.to_euler(code)
        b = Quaternion.from_euler(*euler, axes=code)

    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            self.assertAlmostEqual(a.values[i,j], b.values[i,j], delta=DEL)

    # Quaternion to Matrix3 to Euler and back
    N = 100
    for code in Quaternion._AXES2TUPLE.keys():
        a = Quaternion(np.random.rand(N,4)).unit()
        mats = a.to_matrix3()
        euler = mats.to_euler(code)
        b = Quaternion.from_euler(*euler, axes=code)

    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            self.assertAlmostEqual(a.values[i,j], b.values[i,j], delta=DEL)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
