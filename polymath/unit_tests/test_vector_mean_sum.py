################################################################################
# Vector.mean(), Vector.sum() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector

#*******************************************************************************
# Test_Vector_mean_sum
#*******************************************************************************
class Test_Vector_mean_sum(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #------------
    # Mean
    #------------
    self.assertEqual(Vector([1,2,3,4]).mean(), [1,2,3,4])

    vals = np.random.randn(5,4)
    v = Vector(vals)
    self.assertEqual(v.mean(), np.mean(vals, axis=0))
    self.assertEqual(v.mean(axis=-1), np.mean(vals, axis=0))

    vals = np.random.randn(5,5,4)
    v = Vector(vals)
    self.assertEqual(v.mean(), np.mean(vals, axis=(0,1)))
    self.assertEqual(v.mean(axis=0), np.mean(vals, axis=0))
    self.assertEqual(v.mean(axis=-2), np.mean(vals, axis=0))
    self.assertEqual(v.mean(axis=1), np.mean(vals, axis=1))
    self.assertEqual(v.mean(axis=-1), np.mean(vals, axis=1))

    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    v = Vector(vals, mask)
    self.assertEqual(v.mean(), np.mean(vals[:,:2], axis=(0,1)))
    self.assertEqual(v.mean(axis=1), np.mean(vals[:,:2], axis=1))
    self.assertEqual(v.mean(axis=-1), np.mean(vals[:,:2], axis=1))
    self.assertEqual(v.mean(axis=0)[:2], np.mean(vals[:,:2], axis=0))
    self.assertEqual(np.all(v.mean(axis=0)[2:].mask), True)

    #----------------------
    # Mean, with derivs
    #----------------------
    vals = np.random.randn(5,4)
    dv_dt = Vector(np.random.randn(5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    self.assertEqual(v.mean(), np.mean(vals, axis=0))
    self.assertEqual(v.mean(axis=-1), np.mean(vals, axis=0))
    self.assertEqual(v.mean().d_dt, np.mean(dv_dt.vals, axis=0))
    self.assertEqual(v.mean(axis=-1).d_dt, np.mean(dv_dt.vals, axis=0))

    vals = np.random.randn(5,5,4)
    dv_dt = Vector(np.random.randn(5,5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    self.assertEqual(v.mean(), np.mean(vals, axis=(0,1)))
    self.assertEqual(v.mean(axis=0), np.mean(vals, axis=0))
    self.assertEqual(v.mean(axis=-2), np.mean(vals, axis=0))
    self.assertEqual(v.mean(axis=1), np.mean(vals, axis=1))
    self.assertEqual(v.mean(axis=-1), np.mean(vals, axis=1))
    self.assertEqual(v.mean().d_dt, np.mean(dv_dt.vals, axis=(0,1)))
    self.assertEqual(v.mean(axis=0).d_dt, np.mean(dv_dt.vals, axis=0))
    self.assertEqual(v.mean(axis=-2).d_dt, np.mean(dv_dt.vals, axis=0))
    self.assertEqual(v.mean(axis=1).d_dt, np.mean(dv_dt.vals, axis=1))
    self.assertEqual(v.mean(axis=-1).d_dt, np.mean(dv_dt.vals, axis=1))

    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    dv_dt = Vector(np.random.randn(3,5,4,2,2), drank=2, mask=mask)
    v = Vector(vals, mask, derivs={'t': dv_dt})
    self.assertEqual(v.mean(), np.mean(vals[:,:2], axis=(0,1)))
    self.assertEqual(v.mean(axis=1), np.mean(vals[:,:2], axis=1))
    self.assertEqual(v.mean(axis=-1), np.mean(vals[:,:2], axis=1))
    self.assertEqual(v.mean(axis=0)[:2], np.mean(vals[:,:2], axis=0))
    self.assertEqual(np.all(v.mean(axis=0)[2:].mask), True)

    self.assertEqual(v.mean().d_dt, np.mean(dv_dt.vals[:,:2], axis=(0,1)))
    self.assertEqual(v.mean(axis=1).d_dt, np.mean(dv_dt.vals[:,:2], axis=1))
    self.assertEqual(v.mean(axis=-1).d_dt, np.mean(dv_dt.vals[:,:2], axis=1))
    self.assertEqual(v.mean(axis=0)[:2].d_dt, np.mean(dv_dt.vals[:,:2], axis=0))
    self.assertEqual(np.all(v.mean(axis=0)[2:].d_dt.mask), True)

    #------------
    # Sum
    #------------
    self.assertEqual(Vector([1,2,3,4]).sum(), [1,2,3,4])

    vals = np.random.randn(5,4)
    v = Vector(vals)
    self.assertEqual(v.sum(), np.sum(vals, axis=0))
    self.assertEqual(v.sum(axis=-1), np.sum(vals, axis=0))

    vals = np.random.randn(5,5,4)
    v = Vector(vals)
    self.assertEqual(v.sum(), np.sum(vals, axis=(0,1)))
    self.assertEqual(v.sum(axis=0), np.sum(vals, axis=0))
    self.assertEqual(v.sum(axis=-2), np.sum(vals, axis=0))
    self.assertEqual(v.sum(axis=1), np.sum(vals, axis=1))
    self.assertEqual(v.sum(axis=-1), np.sum(vals, axis=1))

    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    v = Vector(vals, mask)
    self.assertEqual(v.sum(), np.sum(vals[:,:2], axis=(0,1)))
    self.assertEqual(v.sum(axis=1), np.sum(vals[:,:2], axis=1))
    self.assertEqual(v.sum(axis=-1), np.sum(vals[:,:2], axis=1))
    self.assertEqual(v.sum(axis=0)[:2], np.sum(vals[:,:2], axis=0))
    self.assertEqual(np.all(v.sum(axis=0)[2:].mask), True)

    #---------------------
    # Sum, with derivs
    #---------------------
    vals = np.random.randn(5,4)
    dv_dt = Vector(np.random.randn(5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    self.assertEqual(v.sum(), np.sum(vals, axis=0))
    self.assertEqual(v.sum(axis=-1), np.sum(vals, axis=0))
    self.assertEqual(v.sum().d_dt, np.sum(dv_dt.vals, axis=0))
    self.assertEqual(v.sum(axis=-1).d_dt, np.sum(dv_dt.vals, axis=0))

    vals = np.random.randn(5,5,4)
    dv_dt = Vector(np.random.randn(5,5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    self.assertEqual(v.sum(), np.sum(vals, axis=(0,1)))
    self.assertEqual(v.sum(axis=0), np.sum(vals, axis=0))
    self.assertEqual(v.sum(axis=-2), np.sum(vals, axis=0))
    self.assertEqual(v.sum(axis=1), np.sum(vals, axis=1))
    self.assertEqual(v.sum(axis=-1), np.sum(vals, axis=1))
    self.assertEqual(v.sum().d_dt, np.sum(dv_dt.vals, axis=(0,1)))
    self.assertEqual(v.sum(axis=0).d_dt, np.sum(dv_dt.vals, axis=0))
    self.assertEqual(v.sum(axis=-2).d_dt, np.sum(dv_dt.vals, axis=0))
    self.assertEqual(v.sum(axis=1).d_dt, np.sum(dv_dt.vals, axis=1))
    self.assertEqual(v.sum(axis=-1).d_dt, np.sum(dv_dt.vals, axis=1))

    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    dv_dt = Vector(np.random.randn(3,5,4,2,2), drank=2, mask=mask)
    v = Vector(vals, mask, derivs={'t': dv_dt})
    self.assertEqual(v.sum(), np.sum(vals[:,:2], axis=(0,1)))
    self.assertEqual(v.sum(axis=1), np.sum(vals[:,:2], axis=1))
    self.assertEqual(v.sum(axis=-1), np.sum(vals[:,:2], axis=1))
    self.assertEqual(v.sum(axis=0)[:2], np.sum(vals[:,:2], axis=0))
    self.assertEqual(np.all(v.sum(axis=0)[2:].mask), True)

    self.assertEqual(v.sum().d_dt, np.sum(dv_dt.vals[:,:2], axis=(0,1)))
    self.assertEqual(v.sum(axis=1).d_dt, np.sum(dv_dt.vals[:,:2], axis=1))
    self.assertEqual(v.sum(axis=-1).d_dt, np.sum(dv_dt.vals[:,:2], axis=1))
    self.assertEqual(v.sum(axis=0)[:2].d_dt, np.sum(dv_dt.vals[:,:2], axis=0))
    self.assertEqual(np.all(v.sum(axis=0)[2:].d_dt.mask), True)
  #=============================================================================




#*******************************************************************************




################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
