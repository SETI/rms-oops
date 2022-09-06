################################################################################
# Qube.any() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Boolean, Units

class Test_Qube_any(unittest.TestCase):

  # setUp
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  # tearDown
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  # runTest
  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(0.3).any(), True)
    self.assertEqual(type(Scalar(0.3).any()), bool)

    self.assertEqual(Scalar(0.).any(), False)
    self.assertEqual(type(Scalar(0.).any()), bool)

    self.assertEqual(Scalar(4, mask=True).any(), Boolean.MASKED)
    self.assertEqual(type(Scalar(4, mask=True).any()), Boolean)

    # Multiple values
    self.assertTrue(Scalar((0,0,1)).any() == True)
    self.assertEqual(type(Scalar((0,0,1)).any()), bool)

    self.assertEqual(Scalar((1.,2.,3.), True).any(), Boolean.MASKED)
    self.assertEqual(type(Scalar((1.,2.,3.), True).any()), Boolean)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.any(), np.any(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(type(random.any()), bool)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(type(random.any()), bool)

    values = np.random.randn(10)
    random = Scalar(values, mask=True, units=None)
    self.assertEqual(random.any(), Boolean.MASKED)
    self.assertEqual(random.any().units, None)
    self.assertEqual(type(random.any()), Boolean)

    # Test derivs
    values = np.random.randn(10)
    d_dt = Scalar(np.random.randn(10))
    random = Scalar(values)
    random.insert_deriv('t', d_dt)
    self.assertEqual(type(random.any()), bool)

    # Masks
    x = Scalar([0,1,2,3])
    self.assertTrue(x.any())

    x = Scalar(x.values, mask=[False,True,True,True])
    self.assertFalse(x.any())

    x = Scalar(x.values, mask=[True,True,True,True])
    self.assertEqual(x.any(), Boolean.MASKED)

    # Any() over axes
    values = np.zeros(30).reshape(2,3,5) % 16
    values[0,0,0] = 1
    values[1,1,1] = 1
    x = Scalar(values)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.any(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.any(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), bool)
    self.assertEqual(m012, True)

    # Any() with masks
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True

    x = Scalar(values, mask)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    xx = x.values.copy()
    xx[mask] = False
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.any(xx[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01, [False, True, False, False, False])
    self.assertEqual(m012, True)

    mask[:,0] = True
    x = Scalar(values, mask)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))

    for j in (1,2):
      for k in range(5):
        if (j,k) == (0,0):
            continue
        self.assertEqual(m0[j,k], np.any(x.values[:,j,k]))

    j = 0
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
#         self.assertTrue(np.any(m0[j,k].values == np.any(x.values[:,j,k])))
# Changed 3/14. No need to set values where masked

    x = Scalar(values, True)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))

    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], Boolean.MASKED)

    for k in range(5):
        self.assertEqual(m01[k], Boolean.MASKED)

    self.assertEqual(m012, Boolean.MASKED)

    # Qube.tvl_any() tests
    x = Boolean([True, True, True, True])
    self.assertEqual(x.any(), True)
    self.assertEqual(x.tvl_any(), True)

    x = Boolean([False, False, False, False], [False, False, False, False])
    self.assertEqual(x.any(), False)
    self.assertEqual(x.tvl_any(), False)

    x = Boolean([False, False, False, True], [False, False, False, False])
    self.assertEqual(x.any(), True)
    self.assertEqual(x.tvl_any(), True)

    x = Boolean([False, False, False, True], [False, False, False, True])
    self.assertEqual(x.any(), False)
    self.assertEqual(x.tvl_any(), Boolean.MASKED)

    x = Boolean([True, False, False, True], [False, False, False, True])
    self.assertEqual(x.any(), True)
    self.assertEqual(x.tvl_any(), True)

    x = Boolean([False, True, True], True)
    self.assertEqual(x.any(), Boolean.MASKED)
    self.assertEqual(x.tvl_any(), Boolean.MASKED)

    x = Boolean([False, True, True], [True, True, True])
    self.assertEqual(x.any(), Boolean.MASKED)
    self.assertEqual(x.tvl_any(), Boolean.MASKED)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
