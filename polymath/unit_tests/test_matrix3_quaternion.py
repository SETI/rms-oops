################################################################################
# Tests for Matrix3.to_quaternion()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix3, Quaternion

class Test_Matrix3_quaternion(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(4851)

    N = 100
    q = Quaternion(np.random.randn(N,4)).unit()

    mats = Matrix3.as_matrix3(q)
    q2 = mats.to_quaternion()

    DEL = 3.e-14
    for i in range(N):
        # The sign of the whole quaternion might be reversed.
        self.assertTrue(min((q[i] - q2[i]).rms(), (q[i] + q2[i]).rms()) < DEL)

    ########################
    # Test derivatives
    ########################

    N = 100
    q = Quaternion(np.random.randn(N,4)).unit()
    q.insert_deriv('t', Quaternion(np.random.randn(N,4)))

    m = Matrix3.as_matrix3(q, recursive=True)
    self.assertTrue(hasattr(m, 'd_dt'))
    q2 = Matrix3.to_quaternion(m)

    DEL = 1.e-14
    for i in range(N):
        # The sign of the whole quaternion might be reversed.
        self.assertTrue(min((q[i] - q2[i]).rms(), (q[i] + q2[i]).rms()) < DEL)

    EPS = 1.e-6
    dq = q.d_dt * EPS
    q_prime = q.wod + dq
    m_prime = Matrix3.as_matrix3(q_prime)

    dm = m_prime - m

    DEL = 1.e-4
    for i in range(N):
        self.assertTrue((dm[i]/EPS - m.d_dt[i]).rms() < DEL)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
