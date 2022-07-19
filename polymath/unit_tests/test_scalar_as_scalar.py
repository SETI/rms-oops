################################################################################
# Scalar.as_scalar() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Vector, Boolean, Units

#*******************************************************************************
# Test_Scalar_as_scalar
#*******************************************************************************
class Test_Scalar_as_scalar(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    N = 10
    a = Scalar(np.random.randn(N))
    da_dt = Scalar(np.random.randn(N,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Scalar.as_scalar(a, recursive=False)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))

    #-----------------
    # Units case          
    #-----------------
    a = Units.CM
    b = Scalar.as_scalar(a)
    self.assertTrue(type(b), Scalar)
    self.assertEqual(b.units, Units.CM)
    self.assertEqual(b.shape, ())
    self.assertEqual(b.numer, ())
    self.assertEqual(b.values, 1.e-5)

    #--------------------------
    # Vector case is invalid       
    #--------------------------
    a = Vector(np.random.randn(N,3))
    self.assertRaises(ValueError, Scalar.as_scalar, a)

    #------------------
    # Boolean case         
    #------------------
    a = Boolean(np.random.randn(N) < 0.)
    b = Scalar.as_scalar(a)
    self.assertTrue(type(b), Scalar)
    self.assertEqual(b.units, None)
    self.assertEqual(b.shape, (N,))
    self.assertEqual(b.numer, ())
    self.assertEqual(b, a)

    b = Scalar.as_scalar(Boolean(True))
    self.assertTrue(type(b), Scalar)
    self.assertEqual(b.units, None)
    self.assertEqual(b.shape, ())
    self.assertEqual(b.numer, ())
    self.assertEqual(b.values, 1)

    #-------------------
    # Other cases    
    #-------------------
    b = Scalar.as_scalar(3.14159)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.units is None)
    self.assertEqual(b.shape, ())
    self.assertEqual(b.numer, ())
    self.assertEqual(b, 3.14159)

    a = np.arange(120).reshape((2,4,3,5))
    b = Scalar.as_scalar(a)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.units is None)
    self.assertEqual(b.shape, (2,4,3,5))
    self.assertEqual(b.numer, ())
    self.assertEqual(b, a)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
