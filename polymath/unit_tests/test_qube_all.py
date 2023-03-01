################################################################################
# Qube.all() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Boolean, Units

class Test_Qube_all(unittest.TestCase):

  # setUp
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  # tearDown
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  # runTest
  def runTest(self):

    np.random.seed(7456)

    # Individual values
    self.assertEqual(Scalar(0.3).all(), True)
    self.assertEqual(type(Scalar(0.3).all()), bool)

    self.assertEqual(Scalar(0.).all(), False)
    self.assertEqual(type(Scalar(0.).all()), bool)

    self.assertEqual(Scalar(4, mask=True).all(), Boolean.MASKED)
    self.assertEqual(type(Scalar(4, mask=True).all()), Boolean)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).all() == True)
    self.assertEqual(type(Scalar((1,2,3)).all()), bool)

    self.assertTrue(Scalar((0., 1.,2.,3.)).all() == False)
    self.assertEqual(type(Scalar((0., 1.,2.,3.)).all()), bool)

    self.assertEqual(Scalar((1.,2.,3.), True).all(), Boolean.MASKED)
    self.assertEqual(type(Scalar((1.,2.,3.), True).all()), Boolean)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.all(), np.all(x.values))

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(type(random.all()), bool)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(type(random.all()), bool)

    values = np.random.randn(10)
    random = Scalar(values, mask=True, units=None)
    self.assertEqual(random.all(), Boolean.MASKED)
    self.assertEqual(random.all().units, None)
    self.assertEqual(type(random.all()), Boolean)

    # Test derivs
    values = np.random.randn(10)
    d_dt = Scalar(np.random.randn(10))
    random = Scalar(values)
    random.insert_deriv('t', d_dt)
    self.assertEqual(type(random.all()), bool)

    # Masks
    x = Scalar([0,1,2,3])
    self.assertFalse(x.all())

    x = Scalar(x.values, mask=[True,False,False,False])
    self.assertTrue(x.all())

    x = Scalar(x.values, mask=[True,True,True,True])
    self.assertEqual(x.all(), Boolean.MASKED)

    # All() over axes
    x = Scalar(np.arange(30).reshape(2,3,5) % 16)
    m0 = x.all(axis=0)
    m01 = x.all(axis=(0,1))
    m012 = x.all(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.all(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.all(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), bool)
    self.assertEqual(m012, 0)

    # Maxes with masks
    values = np.arange(30).reshape(2,3,5) % 16
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True

    x = Scalar(values, mask)
    m0 = x.all(axis=0)
    m01 = x.all(axis=(0,1))
    m012 = x.all(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    xx = x.values.copy()
    xx[mask] = 1

    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.all(xx[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01, [True, False, True, True, True])

    self.assertEqual(m012, False)

    values = np.arange(30).reshape(2,3,5) % 16
    mask = np.zeros((2,3,5), dtype='bool')
    mask[:,1] = True

    x = Scalar(values, mask)
    m0 = x.all(axis=0)

    for j in (0,2):
      for k in range(5):
        self.assertEqual(m0[j,k], np.all(x.values[:,j,k]))

    j = 1
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
        self.assertTrue(np.all(m0[j,k].values == np.all(x.values[:,j,k])))

    x = Scalar(values, True)
    m0 = x.all(axis=0)
    m01 = x.all(axis=(0,1))
    m012 = x.all(axis=(-1,1,0))

    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], Boolean.MASKED)

    for k in range(5):
        self.assertEqual(m01[k], Boolean.MASKED)

    self.assertEqual(m012, Boolean.MASKED)

    # Qube.tvl_all() tests
    x = Boolean([True, True, True, True])
    self.assertEqual(x.all(), True)
    self.assertEqual(x.tvl_all(), True)

    x = Boolean([True, True, True, True], [False, False, False, False])
    self.assertEqual(x.all(), True)
    self.assertEqual(x.tvl_all(), True)

    x = Boolean([True, True, True, True], [False, False, False, True])
    self.assertEqual(x.all(), True)
    self.assertEqual(x.tvl_all(), Boolean.MASKED)

    x = Boolean([False, True, True], [False, False, False])
    self.assertEqual(x.all(), False)
    self.assertEqual(x.tvl_all(), False)

    x = Boolean([False, True, True], [False, True, True])
    self.assertEqual(x.all(), False)
    self.assertEqual(x.tvl_all(), False)

    x = Boolean([False, True, True], [True, True, True])
    self.assertEqual(x.all(), Boolean.MASKED)
    self.assertEqual(x.tvl_all(), Boolean.MASKED)

    x = Boolean([False, True, True], [True, False, True])
    self.assertEqual(x.all(), True)
    self.assertEqual(x.tvl_all(), Boolean.MASKED)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
