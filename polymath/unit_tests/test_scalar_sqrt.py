################################################################################
# Scalar.sqrt() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

#*******************************************************************************
# Test_Scalar_sqrt
#*******************************************************************************
class Test_Scalar_sqrt(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-----------------------
    # Individual values     
    #-----------------------
    self.assertEqual(Scalar(0.3).sqrt(), np.sqrt(0.3))
    self.assertEqual(type(Scalar(0.3).sqrt()), Scalar)

    self.assertEqual(Scalar(4.).sqrt(), np.sqrt(4.))
    self.assertEqual(Scalar(4).sqrt(), 2.)

    #---------------------
    # Multiple values      
    #---------------------
    self.assertEqual(Scalar((1,2,3)).sqrt(), np.sqrt((1,2,3)))
    self.assertEqual(type(Scalar((1,2,3)).sqrt()), Scalar)

    #------------
    # Arrays     
    #------------
    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.sqrt()
    for i in range(N):
        if x.values[i] >= 0.:
            self.assertEqual(y[i], np.sqrt(x.values[i]))
            self.assertFalse(y.mask[i])
        else:
            self.assertTrue(y.mask[i])

    for i in range(N-1):
        if np.all(x.values[i:i+2] >= 0):
            self.assertEqual(y[i:i+2], np.sqrt(x.values[i:i+2]))

    #----------------------
    # Test valid units       
    #----------------------
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.sqrt, random)

    random = Scalar((4.,9.,16.), units=Units.KM**2)
    self.assertEqual(random.sqrt(), (2,3,4))
    self.assertEqual(random.sqrt(), Scalar((2,3,4), units=Units.KM))

    values = np.random.randn(10)
    random = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, Scalar.sqrt, random)

    random = Scalar(values, units=Units.DEG)
    self.assertRaises(ValueError, Scalar.sqrt, random)

    random = Scalar(values, units=Units.RAD)
    self.assertRaises(ValueError, Scalar.sqrt, random)

    x = Scalar(4., units=Units.UNITLESS)
    self.assertFalse(x.sqrt().mask)

    x = Scalar(-4., units=Units.UNITLESS)
    self.assertTrue(x.sqrt().mask)

    #------------
    # Masks     
    #------------
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.sqrt()
    self.assertTrue(np.all(y.mask[x.mask]))

    #-----------------
    # Derivatives         
    #-----------------
    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))

    self.assertIn('t', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))

    self.assertIn('t', x.sqrt().derivs)
    self.assertTrue(hasattr(x.sqrt(), 'd_dt'))

    EPS = 1.e-6
    y1 = (x + EPS).sqrt()
    y0 = (x - EPS).sqrt()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.sqrt().d_dt

    DEL = 1.e-5
    for i in range(N):
        self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], dy_dt[i],
                               delta = abs(dy_dt[i]) * DEL)

    #-----------------------------------------------
    # Read-only status should NOT be preserved    
    #-----------------------------------------------
    N = 10
    x = Scalar(np.random.randn(N))
    self.assertFalse(x.readonly)
    self.assertFalse(x.sqrt().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().sqrt().readonly)

    ###### Without Checking
    N = 1000
    x = Scalar(np.random.randn(N))
    self.assertRaises(ValueError, x.sqrt, check=False)

    x = Scalar(np.random.randn(N).clip(0,1.e308))
    self.assertEqual(x.sqrt(), np.sqrt(x.values))
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
