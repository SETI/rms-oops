################################################################################
# Tests for Qube.__pow__()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_power(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(9947)

    a = Matrix(np.random.randint(-100, 101, (10,5,2,2)))

    self.assertEqual(a**0, a.identity())
    self.assertEqual(a**1, a)
    self.assertEqual(a**2, a*a)
    self.assertEqual(a**3, a*a*a)
    self.assertEqual(a**4, a*a*a*a)
    self.assertEqual(a**5, a*a*a*a*a)
    self.assertEqual(a**6, a*a*a*a*a*a)

    self.assertTrue(np.all(abs((a**7 ).vals - (a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**8 ).vals - (a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**9 ).vals - (a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**10).vals - (a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**11).vals - (a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**12).vals - (a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**13).vals - (a*a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**14).vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**15).vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)

    b = a.inverse()
    self.assertEqual(a**-1, b)
    self.assertTrue(np.all(abs((a**-2 ).vals - (b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-3 ).vals - (b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-4 ).vals - (b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-5 ).vals - (b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-6 ).vals - (b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-7 ).vals - (b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-8 ).vals - (b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-9 ).vals - (b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-10).vals - (b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-11).vals - (b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-12).vals - (b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-13).vals - (b*b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-14).vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-15).vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)

    a.insert_deriv('t', Matrix(np.random.randn(10,5,2,2)))

    self.assertTrue(np.all((a**0).d_dt.vals == 0.))
    self.assertEqual((a**1).d_dt, a.d_dt)

    self.assertTrue(np.all(abs((a**2 ).d_dt.vals - (a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**3 ).d_dt.vals - (a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**4 ).d_dt.vals - (a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**5 ).d_dt.vals - (a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**6 ).d_dt.vals - (a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**7 ).d_dt.vals - (a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**8 ).d_dt.vals - (a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**9 ).d_dt.vals - (a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**10).d_dt.vals - (a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**11).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**12).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**13).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**14).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**15).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)

    b = a.inverse()

    self.assertTrue(np.all(abs((a**-1 ).d_dt.vals - (b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-2 ).d_dt.vals - (b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-3 ).d_dt.vals - (b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-4 ).d_dt.vals - (b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-5 ).d_dt.vals - (b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-6 ).d_dt.vals - (b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-7 ).d_dt.vals - (b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-8 ).d_dt.vals - (b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-9 ).d_dt.vals - (b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-10).d_dt.vals - (b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-11).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-12).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-13).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-14).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    self.assertTrue(np.all(abs((a**-15).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)

############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
