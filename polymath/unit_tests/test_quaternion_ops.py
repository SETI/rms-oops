################################################################################
# Tests for Quaternion arithmetic operations
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Quaternion, Matrix3

class Test_Quaternion_ops(unittest.TestCase):

  def runTest(self):


    N = 3
    M = 2
    a = Quaternion(np.random.randn(N,1,4))
    a.insert_deriv('t', Quaternion(np.random.randn(N,1,4,2), drank=1))

    b = Quaternion(np.random.randn(M,4))
    b.insert_deriv('t', Quaternion(np.random.randn(M,4,2), drank=1))

    self.assertEqual(a, a * Quaternion.IDENTITY)
    self.assertEqual(a, a / Quaternion.IDENTITY)

    self.assertEqual(a, a + Quaternion.ZERO)
    self.assertEqual(a, a - Quaternion.ZERO)

    # Multiply...
    (sa,va) = a.to_parts()
    (sb,vb) = b.to_parts()

    # Formula from http://en.wikipedia.org/wiki/Quaternion
    sab = sa * sb - va.dot(vb)
    vab = sa * vb + sb * va + va.cross(vb)

    ab = Quaternion.from_parts(sab, vab)

    DEL = 1.e-14
    self.assertTrue((ab - a*b).rms().max() < DEL)

    dab_dt = a.without_derivs() * b.d_dt + a.d_dt * b.without_derivs()
    self.assertTrue((dab_dt - (a*b).d_dt).rms().max() < DEL)

    # Divide...
    test = ab / b
    self.assertTrue((test - a).rms().max() < DEL)

    b_inv = b.reciprocal()
    test = ab * b_inv
    self.assertTrue((test - a).rms().max() < DEL)

    ab_wod = ab.without_derivs()
    b_inv_wod = b_inv.without_derivs()

    dtest_dt = ab.d_dt * b_inv_wod + ab_wod * b_inv.d_dt
    self.assertTrue((dtest_dt - a.d_dt).rms().max() < DEL)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
