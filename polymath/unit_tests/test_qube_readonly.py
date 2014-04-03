################################################################################
# Tests for Qube methods related to read-only and read-write status
#
#   as_readonly(self, nocopy='')
#   copy(self, readonly=False, recursive=True)
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_readonly(unittest.TestCase):

  def runTest(self):

    a = Vector(np.random.randn(4,5,6,3,2), drank=1)
    self.assertEqual(a.readonly, False)
    a.values[0,0,0,0,0] = 1.

    a = a.as_readonly()
    self.assertEqual(a.readonly, True)

    self.assertRaises((ValueError,RuntimeError), a.values.__setitem__,
                      (0,0,0,0,0), 1.)

    ####
    a = Vector(np.random.randn(4,5,6,3,2), drank=1).as_readonly()
    b = a.copy()
    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, False)

    a = Vector(np.random.randn(4,5,6,3,2), drank=1).as_readonly()
    b = a.copy(readonly=True)
    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)
    self.assertTrue(a is b)

    ####
    a = Vector(np.random.randn(5,3))
    da_dm = Vector(np.random.randn(5,3,2,3), drank=2)
    a.insert_deriv('m', da_dm)
    self.assertEqual(a.readonly, False)
    self.assertEqual(a.d_dm.readonly, False)

    b = a.copy(readonly=True, recursive=False)
    self.assertEqual(b.readonly, True)
    self.assertFalse(hasattr(b, 'd_dm'))

    b = a.copy(readonly=False, recursive=True)
    self.assertEqual(b.readonly, False)
    self.assertEqual(b.d_dm.readonly, False)

    b = a.copy(readonly=True, recursive=True)
    self.assertEqual(b.readonly, True)
    self.assertEqual(b.d_dm.readonly, True)

    ####
    a = Vector(np.random.randn(5,3))
    da_dm = Vector(np.random.randn(5,3,2,3), drank=2)
    a.insert_deriv('m', da_dm)
    self.assertEqual(a.readonly, False)
    self.assertEqual(a.d_dm.readonly, False)

    b = a.copy()
    self.assertTrue(np.all(a.values == b.values))

    b.values[0,0] = 42
    self.assertTrue(a.values[0,0] != 42)

    b.d_dm.values[0,0,0,0] = 42
    self.assertTrue(a.d_dm.values[0,0,0,0] != 42)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
