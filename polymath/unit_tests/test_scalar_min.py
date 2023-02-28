################################################################################
# Scalar.min() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_min(unittest.TestCase):

  # setUp
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  # tearDown
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  # runTest
  def runTest(self):

    np.random.seed(2956)

    # Individual values
    self.assertEqual(Scalar(0.3).min(), 0.3)
    self.assertEqual(type(Scalar(0.3).min()), float)

    self.assertEqual(Scalar(4).min(), 4)
    self.assertEqual(type(Scalar(4).min()), int)

    self.assertTrue(Scalar(4, mask=True).min().mask)
    self.assertEqual(type(Scalar(4, mask=True).min()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).min() == 1)
    self.assertEqual(type(Scalar((1,2,3)).min()), int)

    self.assertTrue(Scalar((1.,2.,3.)).min() == 1.)
    self.assertEqual(type(Scalar((1.,2,3)).min()), float)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.min(), np.min(x.values))

    argmin = x.argmin()
    self.assertEqual(x.flatten()[argmin], x.min())

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.min().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.min().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.min()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    minval = np.inf
    for i in range(N):
        if (not x.mask[i]) and (x.values[i] < minval):
            minval = x.values[i]

    self.assertEqual(minval, x.min())

    argmin = x.argmin()
    self.assertEqual(x[argmin], x.min())

    # If we mask the minimum value(s), the minimum should increase
    x = x.mask_where_eq(minval)
    self.assertTrue(x.min() > minval)

    argmin = x.argmin()
    self.assertEqual(x.flatten()[argmin], x.min())

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.min().mask)
    self.assertTrue(type(masked.min()), Scalar)

    argmin = x.argmin()
    self.assertEqual(x[argmin], x.min())

    # Denominators
    a = Scalar([1.,2.], drank=1)
    self.assertRaises(ValueError, a.min)

    # Mins over axes
    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.min(axis=0)
    m01 = x.min(axis=(0,1))
    m012 = x.min(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.min(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.min(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), int)
    self.assertEqual(m012, 0)

    argmin = x.argmin(axis=0)
    for j in range(3):
      for k in range(5):
        self.assertEqual(x[argmin[j,k],j,k], m0[j,k])

    # Mins with masks
    values = np.arange(30).reshape(2,3,5)
    mask = (values < 5)
    x = Scalar(values, mask)
    m0 = x.min(axis=0)
    m01 = x.min(axis=(0,1))
    m012 = x.min(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    xx = x.values.copy()
    xx[xx < 5] += 100
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.min(xx[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01, [5,6,7,8,9])

    self.assertEqual(m012, 5)

    argmin = x.argmin(axis=0)
    for j in range(3):
      for k in range(5):
        self.assertEqual(x[argmin[j,k],j,k], m0[j,k])

    values = np.arange(30).reshape(2,3,5)
    mask = (values < 5)
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.min(axis=0)

    for j in (0,2):
      for k in range(5):
        self.assertEqual(m0[j,k], np.min(xx[:,j,k]))

    j = 1
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
        self.assertTrue(np.all(m0[j,k].values == np.min(x.values[:,j,k])))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
