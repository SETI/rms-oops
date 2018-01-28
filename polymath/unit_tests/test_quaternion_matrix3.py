################################################################################
# Tests for Quaternion.to_matrix3() and Quaternion.from_matrix3()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Quaternion, Matrix3, Matrix

class Test_Quaternion_matrix3(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # Quaternion to Matrix3 and back
    ############################################################################

    # One quaternion
    a = Quaternion(np.random.rand(4)).unit()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)

    DEL = 1.e-14
    for j in range(4):
        self.assertAlmostEqual(a.values[j], b.values[j], delta=DEL)

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = a.as_readonly()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)

    self.assertFalse(b.readonly)

    # N Quaternions
    N = 100
    a = Quaternion(np.random.rand(N,4)).unit()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)

    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            self.assertAlmostEqual(a.values[i,j], b.values[i,j], delta=DEL)

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = a.as_readonly()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)

    self.assertFalse(b.readonly)

    ############################################################################
    # Quaternion to Euler angles and back
    ############################################################################

    # N Quaternions, without unit()
    N = 100
    a = Quaternion(np.random.rand(N,4))
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)

    aa = a.unit()
    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            self.assertAlmostEqual(aa.values[i,j], b.values[i,j], delta=DEL)

    self.assertFalse(aa.readonly)
    self.assertFalse(b.readonly)

    # N Quaternions, with unit()
    N = 100
    a = Quaternion(np.random.rand(N,4)).unit()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)

    aa = a.unit()
    DEL = 2.e-14
    for i in range(N):
        for j in range(4):
            self.assertAlmostEqual(a.values[i,j], b.values[i,j], delta=DEL)

    self.assertFalse(aa.readonly)
    self.assertFalse(b.readonly)

    ############################################################################
    # Quaternion to Matrix3, with derivatives
    ############################################################################

    N = 100
    x = Quaternion(np.random.rand(N,4))
    x.insert_deriv('t', Quaternion((np.random.rand(N,4))))
    y = x.to_matrix3(recursive=True)

    EPS = 1.e-6
    y1 = Matrix.as_matrix((x + (EPS,0,0,0)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (EPS,0,0,0)).to_matrix3(recursive=False))
    dy_dx0 = 0.5 * (y1 - y0) / EPS

    y1 = Matrix.as_matrix((x + (0,EPS,0,0)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (0,EPS,0,0)).to_matrix3(recursive=False))
    dy_dx1 = 0.5 * (y1 - y0) / EPS

    y1 = Matrix.as_matrix((x + (0,0,EPS,0)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (0,0,EPS,0)).to_matrix3(recursive=False))
    dy_dx2 = 0.5 * (y1 - y0) / EPS

    y1 = Matrix.as_matrix((x + (0,0,0,EPS)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (0,0,0,EPS)).to_matrix3(recursive=False))
    dy_dx3 = 0.5 * (y1 - y0) / EPS

    dy_dt = (dy_dx0 * x.d_dt.values[...,0] +
             dy_dx1 * x.d_dt.values[...,1] +
             dy_dx2 * x.d_dt.values[...,2] +
             dy_dx3 * x.d_dt.values[...,3])

    DEL = 1.e-5
    for i in range(N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(dy_dt.values[i,j,k], y.d_dt.values[i,j,k],
                                   delta=DEL)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
