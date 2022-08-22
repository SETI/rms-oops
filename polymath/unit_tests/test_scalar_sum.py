################################################################################
# Scalar.mean() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

#*******************************************************************************
# Test_Scalar_sum
#*******************************************************************************
class Test_Scalar_sum(unittest.TestCase):

  #=============================================================================
  # setUp
  #=============================================================================
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  #=============================================================================
  # tearDown
  #=============================================================================
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-----------------------
    # Individual values
    #-----------------------
    self.assertEqual(Scalar(0.3).sum(), 0.3)
    self.assertEqual(type(Scalar(0.3).sum()), float)

    self.assertEqual(Scalar(4).sum(), 4)
    self.assertEqual(type(Scalar(4).sum()), int)

    self.assertTrue(Scalar(4, mask=True).sum().mask)
    self.assertEqual(type(Scalar(4, mask=True).sum()), Scalar)
    self.assertEqual(type(Scalar(4, mask=True).sum()), Scalar)

    #----------------------
    # Multiple values
    #----------------------
    self.assertTrue(Scalar((1,2,3)).sum() == 6)
    self.assertEqual(type(Scalar((1,2,3)).sum()), int)

    self.assertTrue(Scalar((1.,2.,3.)).sum() == 6.)
    self.assertEqual(type(Scalar((1.,2,3)).sum()), float)

    #------------
    # Arrays
    #------------
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.sum(), np.sum(x.values))

    #---------------------------------------------------------
    # Test units
    #---------------------------------------------------------
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.sum().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.sum().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.sum()), float)

    #---------------
    # Masks
    #---------------
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    sumval = 0.
    for i in range(N):
        if not x.mask[i]:
            sumval += x.values[i]

    self.assertTrue(abs((sumval - x.sum()) / sumval) < 1.e-13)

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.sum().mask)
    self.assertTrue(type(masked.sum()), Scalar)

    #------------------
    # Denominators
    #------------------
    a = Scalar(np.arange(24.).reshape(4,3,2), drank=1)
    b = a.sum(axis=1)
    self.assertEqual(b.shape, (4,))
    self.assertEqual(b, Scalar([[6,9],[24,27],[42,45],[60,63]], drank=1))

    #-------------------
    # Sums over axes
    #-------------------
    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.sum(axis=0)
    m01 = x.sum(axis=(0,1))
    m012 = x.sum(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.sum(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.sum(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), int)
    self.assertEqual(m012, np.sum(np.arange(30)))

    #-----------------------
    # Sums with masks
    #-----------------------
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(np.arange(30).reshape(2,3,5), mask)
    m0 = x.sum(axis=0)
    m01 = x.sum(axis=(0,1))
    m012 = x.sum(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    self.assertEqual(m0[0,0], x.values[1,0,0])
    self.assertEqual(m0[1,1], x.values[0,1,1])
    for j in range(3):
      for k in range(5):
        if (j,k) in [(0,0), (1,1)]: continue
        self.assertEqual(m0[j,k], np.sum(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01[0], (np.sum(x.values[:,:,0]) - x.values[0,0,0]))
    self.assertEqual(m01[1], (np.sum(x.values[:,:,1]) - x.values[1,1,1]))
    self.assertEqual(m01[2],  np.sum(x.values[:,:,2]))
    self.assertEqual(m01[3],  np.sum(x.values[:,:,3]))
    self.assertEqual(m01[4],  np.sum(x.values[:,:,4]))

    self.assertEqual(m012, np.sum(x.values) - x.values[0,0,0] - x.values[1,1,1])

    values = np.arange(30).reshape(2,3,5)
    mask[0,0,0] = True
    mask[1,1,1] = True
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.sum(axis=0)

    self.assertEqual(m0[0,0], x.values[1,0,0])
    for j in (0,2):
      for k in range(5):
        if (j,k) in [(0,0), (1,1)]: continue
        self.assertEqual(m0[j,k], np.sum(x.values[:,j,k]))

    j = 1
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
        self.assertTrue(np.all(m0[j,k].values == m0.default))
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
