################################################################################
# Tests for Qube mask_where() methods
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_vector_masking(unittest.TestCase):

  # runTest
  def runTest(self):

    ############################################################################
    # mask_where_component_le(), etc.
    ############################################################################

    a = Vector(np.arange(9).reshape(3,3))   # [[0,1,2],[3,4,5],[6,7,8]]
    mask000 = np.array([False, False, False])
    mask100 = np.array([True , False, False])
    mask110 = np.array([True , True , False])
    mask111 = np.array([True , True , True ])
    mask011 = np.array([False, True , True ])
    mask001 = np.array([False, False, True ])

    self.assertTrue(np.all(a.mask_where_component_le(2,2).mask == mask100))
    self.assertTrue(np.all(a.mask_where_component_le(2,3).mask == mask100))
    self.assertTrue(np.all(a.mask_where_component_le(2,4).mask == mask100))
    self.assertTrue(np.all(a.mask_where_component_le(2,5).mask == mask110))
    self.assertTrue(np.all(a.mask_where_component_le(2,6).mask == mask110))

    self.assertTrue(np.all(a.mask_where_component_lt(2,2).mask == mask000))
    self.assertTrue(np.all(a.mask_where_component_lt(2,3).mask == mask100))
    self.assertTrue(np.all(a.mask_where_component_lt(2,4).mask == mask100))
    self.assertTrue(np.all(a.mask_where_component_lt(2,5).mask == mask100))
    self.assertTrue(np.all(a.mask_where_component_lt(2,6).mask == mask110))

    self.assertTrue(np.all(a.mask_where_component_ge(2,2).mask == mask111))
    self.assertTrue(np.all(a.mask_where_component_ge(2,3).mask == mask011))
    self.assertTrue(np.all(a.mask_where_component_ge(2,4).mask == mask011))
    self.assertTrue(np.all(a.mask_where_component_ge(2,5).mask == mask011))
    self.assertTrue(np.all(a.mask_where_component_ge(2,6).mask == mask001))
    self.assertTrue(np.all(a.mask_where_component_ge(2,7).mask == mask001))

    self.assertTrue(np.all(a.mask_where_component_gt(2,1).mask == mask111))
    self.assertTrue(np.all(a.mask_where_component_gt(2,2).mask == mask011))
    self.assertTrue(np.all(a.mask_where_component_gt(2,3).mask == mask011))
    self.assertTrue(np.all(a.mask_where_component_gt(2,4).mask == mask011))
    self.assertTrue(np.all(a.mask_where_component_gt(2,5).mask == mask001))
    self.assertTrue(np.all(a.mask_where_component_gt(2,6).mask == mask001))
    self.assertTrue(np.all(a.mask_where_component_gt(2,7).mask == mask001))
    self.assertTrue(np.all(a.mask_where_component_gt(2,8).mask == mask000))

    ############################################################################
    # clip_component(), etc.
    ############################################################################

    self.assertEqual(a.clip_component(2,2,8,False), [[0,1,2],[3,4,5],[6,7,8]])
    self.assertEqual(a.clip_component(2,2,7,False), [[0,1,2],[3,4,5],[6,7,7]])
    self.assertEqual(a.clip_component(2,2,6,False), [[0,1,2],[3,4,5],[6,7,6]])
    self.assertEqual(a.clip_component(2,2,3,False), [[0,1,2],[3,4,3],[6,7,3]])
    self.assertEqual(a.clip_component(2,2,None,False), [[0,1,2],[3,4,5],[6,7,8]])
    self.assertEqual(a.clip_component(2,None,3,False), [[0,1,2],[3,4,3],[6,7,3]])

    self.assertEqual(a.clip_component(2,2,8,True), [[0,1,2],[3,4,5],[6,7,8]])
    self.assertTrue(np.all(a.clip_component(2,2,7,True).mask == mask001))
    self.assertTrue(np.all(a.clip_component(2,2,6,True).mask == mask001))
    self.assertTrue(np.all(a.clip_component(2,2,3,True).mask == mask011))
    self.assertTrue(np.all(a.clip_component(2,2,None,True).mask == mask000))

    lower = Scalar([4,3,2])
    upper = Scalar([5,4,3],mask=[0,1,0])
    self.assertEqual(a.clip_component(2,lower,upper,False), [[0,1,4],[3,4,5],[6,7,3]])

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
