################################################################################
# Matrix3.twovec() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix3, Vector3

class Test_Matrix3_twovec(unittest.TestCase):

  def runTest(self):

    DEL = 1.e-12

    # These all regenerate the Identity matrix
    mat = Matrix3.twovec(Vector3.XAXIS, 0, Vector3.YAXIS, 1)
    self.assertTrue((Matrix3.IDENTITY - mat).rms() < DEL)

    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.ZAXIS, 2)
    self.assertTrue((Matrix3.IDENTITY - mat).rms() < DEL)

    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.YAXIS + Vector3.ZAXIS, 2)
    self.assertTrue((Matrix3.IDENTITY - mat).rms() < DEL)

    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.YAXIS + Vector3.ZAXIS, 2)
    self.assertTrue((Matrix3.IDENTITY - mat).rms() < DEL)

    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.ZAXIS - 99*Vector3.YAXIS, 2)
    self.assertTrue((Matrix3.IDENTITY - mat).rms() < DEL)

    # Test random vectors
    N = 100
    a = Vector3(np.random.randn(N,3)).unit()
    mat = Matrix3.twovec(a, 2, Vector3.XAXIS, 0)

    for i in range(N):

        # The new Y-axis is perpendicular to X
        self.assertAlmostEqual(mat.values[i,1,0], 0., delta=DEL)

        # The new Z-axis coincides with the line of sight
        self.assertAlmostEqual(mat.values[i,2,0], a.values[i,0], delta=DEL)
        self.assertAlmostEqual(mat.values[i,2,1], a.values[i,1], delta=DEL)
        self.assertAlmostEqual(mat.values[i,2,2], a.values[i,2], delta=DEL)

    # Test masks
    a = Vector3(np.random.randn(N,3), mask=np.random.randn(N) < -0.5)
    b = Vector3(np.random.randn(N,3), mask=np.random.randn(N) < -0.5)
    mat = Matrix3.twovec(a, 2, b, 1)

    self.assertTrue(np.all(mat.mask == (a.mask | b.mask)))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
