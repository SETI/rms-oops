################################################################################
# Scalar.mean() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_mean(unittest.TestCase):

  # setUp
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  # tearDown
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  # runTest
  def runTest(self):

    np.random.seed(2659)

    # Individual values
    self.assertEqual(Scalar(0.3).mean(), 0.3)
    self.assertEqual(type(Scalar(0.3).mean()), float)

    self.assertEqual(Scalar(4).mean(), 4)
    self.assertEqual(type(Scalar(4).mean()), float)

    self.assertTrue(Scalar(4, mask=True).mean().mask)
    self.assertEqual(type(Scalar(4, mask=True).mean()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).mean() == 2)
    self.assertEqual(type(Scalar((1,2,3)).mean()), float)

    self.assertTrue(Scalar((1,2,3,4)).mean() == 2.5)
    self.assertEqual(type(Scalar((1,2,3,4)).mean()), float)

    self.assertTrue(Scalar((1.,2.,3.)).mean() == 2.)
    self.assertEqual(type(Scalar((1.,2,3)).mean()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.mean(), np.mean(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.mean().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.mean().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.mean()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    meanval = 0.
    count = 0
    for i in range(N):
        if not x.mask[i]:
            count += 1
            meanval += x.values[i]

    meanval /= count
    self.assertTrue(abs((meanval - x.mean()) / meanval) < 5.e-14)

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.mean().mask)
    self.assertTrue(type(masked.mean()), Scalar)

    # Means over axes
    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.mean(axis=0)
    m01 = x.mean(axis=(0,1))
    m012 = x.mean(axis=(-1,1,0))
    self.assertTrue(m0.is_float())
    self.assertTrue(m01.is_float())
    if Qube.PREFER_BUILTIN_TYPES: # pragma: no cover
        self.assertTrue(isinstance(m012, float))
    else: # pragma: no cover
        self.assertTrue(m012.is_float())

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.mean(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.mean(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), float)
    self.assertEqual(m012, np.sum(np.arange(30))/30.)

    # Means with masks
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(np.arange(30).reshape(2,3,5), mask)
    m0 = x.mean(axis=0)
    m01 = x.mean(axis=(0,1))
    m012 = x.mean(axis=(-1,1,0))
    self.assertTrue(m0.is_float())
    self.assertTrue(m01.is_float())
    if Qube.PREFER_BUILTIN_TYPES: # pragma: no cover
        self.assertTrue(isinstance(m012, float))
    else: # pragma: no cover
        self.assertTrue(m012.is_float())

    self.assertEqual(m0.shape, (3,5))
    self.assertEqual(m0[0,0], x.values[1,0,0])
    self.assertEqual(m0[1,1], x.values[0,1,1])
    for j in range(3):
      for k in range(5):
        if (j,k) in [(0,0), (1,1)]:
            continue
        self.assertEqual(m0[j,k], np.mean(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01[0], (np.sum(x.values[:,:,0]) - x.values[0,0,0]) / 5.)
    self.assertEqual(m01[1], (np.sum(x.values[:,:,1]) - x.values[1,1,1]) / 5.)
    self.assertEqual(m01[2],  np.sum(x.values[:,:,2]) / 6.)
    self.assertEqual(m01[3],  np.sum(x.values[:,:,3]) / 6.)
    self.assertEqual(m01[4],  np.sum(x.values[:,:,4]) / 6.)

    values = np.arange(30).reshape(2,3,5)
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.mean(axis=0)

    self.assertEqual(m0[0,0], x.values[1,0,0])
    for j in (0,2):
      for k in range(5):
        if (j,k) in [(0,0), (1,1)]:
            continue
        self.assertEqual(m0[j,k], np.mean(x.values[:,j,k]))

    j = 1
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
        self.assertTrue(np.all(m0[j,k].values == m0.default))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
