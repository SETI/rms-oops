################################################################################
# Pair.as_pair() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Pair, Scalar, Pair, Units

#*******************************************************************************
# Test_Pair_as_pair
#*******************************************************************************
class Test_Pair_as_pair(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    N = 10
    a = Pair(np.random.randn(N,2))
    da_dt = Pair(np.random.randn(N,2))
    a.insert_deriv('t', da_dt)

    b = Pair.as_pair(a, recursive=False)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))

    #----------------------
    # Matrix case, 2x1       
    #----------------------
    a = Matrix(np.random.randn(N,2,1), units=Units.REV)
    da_dt = Matrix(np.random.randn(N,2,1,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Pair.as_pair(a)
    self.assertTrue(type(b), Pair)
    self.assertEqual(a.units, b.units)
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.numer, (2,1))
    self.assertEqual(b.numer, (2,))
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))

    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt.shape, b.shape)
    self.assertEqual(b.d_dt.numer, (2,))
    self.assertEqual(b.d_dt.denom, (6,))
    self.assertTrue(np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel()))

    b = Pair.as_pair(a, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    #---------------------
    # Matrix case, 1x2      
    #---------------------
    a = Matrix(np.random.randn(N,1,2), units=Units.REV)
    da_dt = Matrix(np.random.randn(N,1,2,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Pair.as_pair(a)
    self.assertTrue(type(b), Pair)
    self.assertEqual(a.units, b.units)
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.numer, (1,2))
    self.assertEqual(b.numer, (2,))
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))

    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt.shape, b.shape)
    self.assertEqual(b.d_dt.numer, (2,))
    self.assertEqual(b.d_dt.denom, (6,))
    self.assertTrue(np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel()))

    b = Pair.as_pair(a, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    #--------------------
    # Other cases     
    #--------------------
    b = Pair.as_pair((1,2))
    self.assertTrue(type(b), Pair)
    self.assertTrue(b.units is None)
    self.assertEqual(b.shape, ())
    self.assertEqual(b.numer, (2,))
    self.assertEqual(b, (1,2))

    a = np.arange(120).reshape((5,4,3,2))
    b = Pair.as_pair(a)
    self.assertTrue(type(b), Pair)
    self.assertTrue(b.units is None)
    self.assertEqual(b.shape, (5,4,3))
    self.assertEqual(b.numer, (2,))
    self.assertEqual(b, a)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
