################################################################################
# Scalar.median() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_median(unittest.TestCase):

  # setUp
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  # tearDown
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  # runTest
  def runTest(self):

    np.random.seed(9781)

    # Individual values
    self.assertEqual(Scalar(0.3).median(), 0.3)
    self.assertEqual(type(Scalar(0.3).median()), float)

    self.assertEqual(Scalar(4).median(), 4)
    self.assertEqual(type(Scalar(4).median()), float)

    self.assertTrue(Scalar(4, mask=True).median().mask)
    self.assertEqual(type(Scalar(4, mask=True).median()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).median() == 2)
    self.assertEqual(type(Scalar((1,2,3)).median()), float)

    self.assertTrue(Scalar((1,2,3,4)).median() == 2.5)
    self.assertEqual(type(Scalar((1,2,3,4)).median()), float)

    self.assertTrue(Scalar((1.,2.,3.)).median() == 2.)
    self.assertEqual(type(Scalar((1.,2,3)).median()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.median(), np.median(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.median().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.median().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.median()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    self.assertEqual(x.median(), np.median(x.values[~x.mask]))

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.median().mask)
    self.assertTrue(type(masked.median()), Scalar)

    # Means over axes
    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.median(axis=0)
    m01 = x.median(axis=(0,1))
    m012 = x.median(axis=(-1,1,0))
    self.assertTrue(m0.is_float())
    self.assertTrue(m01.is_float())
    self.assertTrue(isinstance(m012, float))

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.median(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.median(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), float)
    self.assertEqual(m012, np.sum(np.arange(30))/30.)

    # Means with masks
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(np.arange(30).reshape(2,3,5), mask)
    m0 = x.median(axis=0)
    m01 = x.median(axis=(0,1))
    m012 = x.median(axis=(-1,1,0))
    self.assertTrue(m0.is_float())
    self.assertTrue(m01.is_float())
    self.assertTrue(isinstance(m012, float))

    self.assertEqual(m0.shape, (3,5))
    self.assertTrue(m0[0,0] == x.values[1,0,0])
    self.assertEqual(m0[1,1], x.values[0,1,1])
    for j in range(3):
      for k in range(5):
        if (j,k) in [(0,0), (1,1)]:
            continue
        self.assertEqual(m0[j,k], np.median(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01[2], np.median(x.values[:,:,2]))
    self.assertEqual(m01[3], np.median(x.values[:,:,3]))
    self.assertEqual(m01[4], np.median(x.values[:,:,4]))

    indices = (np.array([0,0,1,1,1]), np.array([1,2,0,1,2]),
                                      np.array([0,0,0,0,0]))
    self.assertEqual(m01[0], np.median(x.values[indices]))

    indices = (np.array([0,0,0,1,1]), np.array([0,1,2,0,2]),
                                      np.array([1,1,1,1,1]))
    self.assertEqual(m01[1], np.median(x.values[indices]))

    values = np.arange(30).reshape(2,3,5)
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.median(axis=0)

    self.assertEqual(m0[0,0], x.values[1,0,0])
    for j in (0,2):
      for k in range(5):
        if (j,k) in [(0,0), (1,1)]:
            continue
        self.assertEqual(m0[j,k], np.median(x.values[:,j,k]))

    j = 1
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
        self.assertTrue(np.all(m0[j,k].values == np.median(x.values[:,j,k])))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
