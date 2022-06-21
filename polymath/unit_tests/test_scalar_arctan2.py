################################################################################
# Scalar.arctan2() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

#*******************************************************************************
# Test_Scalar_arctan2
#*******************************************************************************
class Test_Scalar_arctan2(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-----------------------
    # Individual values     	
    #-----------------------
    self.assertEqual(Scalar(1.).arctan2(1.), np.arctan2(1,1))
    self.assertEqual(type(Scalar(1.).arctan2(1.)), Scalar)

    self.assertEqual(Scalar(0.).arctan2(0.), np.arctan2(0,0))
    self.assertEqual(Scalar(0.).arctan2(1.), 0.)

    self.assertAlmostEqual(Scalar( 1.).arctan2( 1.),  0.25 * np.pi, 1.e-15)
    self.assertAlmostEqual(Scalar( 1.).arctan2( 0.),  0.5  * np.pi, 1.e-15)
    self.assertAlmostEqual(Scalar( 1.).arctan2(-1.),  0.75 * np.pi, 1.e-15)
    self.assertAlmostEqual(Scalar( 0.).arctan2(-1.),         np.pi, 1.e-15)
    self.assertAlmostEqual(Scalar(-1.).arctan2(-1.), -0.75 * np.pi, 1.e-15)
    self.assertAlmostEqual(Scalar(-1.).arctan2( 0.), -0.5  * np.pi, 1.e-15)
    self.assertAlmostEqual(Scalar(-1.).arctan2( 1.), -0.25 * np.pi, 1.e-15)

    #---------------------
    # Multiple values	      
    #---------------------
    self.assertTrue(abs(4/np.pi * Scalar(1.).arctan2((1,0,-1)) -
                     (1,2,3)).max() < 1.e-15)

    self.assertTrue(abs(4/np.pi * Scalar(-1.).arctan2((1,0,-1)) -
                     (-1,-2,-3)).max() < 1.e-15)

    self.assertTrue(abs(4/np.pi * Scalar((1,0,-1)).arctan2((1,0,-1)) -
                     (1,0,-3)).max() < 1.e-15)

    self.assertTrue(abs(4/np.pi * Scalar((1,0,-1)).arctan2((1.,)) -
                     (1,0,-1)).max() < 1.e-15)

    #----------------
    # Arrays	     	 
    #----------------
    N = 1000
    y = Scalar(np.random.randn(N))
    x = Scalar(np.random.randn(N))
    angle = y.arctan2(x)
    for i in range(N):
        self.assertEqual(angle[i], np.arctan2(y.values[i], x.values[i]))

    for i in range(N-1):
        self.assertEqual(angle[i:i+2], np.arctan2(y.values[i:i+2],
                                                  x.values[i:i+2]))

    #----------------------
    # Test valid units	       
    #----------------------
    values = np.random.rand
    y = Scalar(values, units=Units.KM)
    x = Scalar(values, units=Units.CM)
    self.assertFalse(np.any(y.arctan2(x).mask))

    values = np.random.randn(10)
    y = Scalar(values, units=Units.KM)
    x = Scalar(values, units=None)
    self.assertFalse(np.any(y.arctan2(x).mask))

    values = np.random.randn(10)
    y = Scalar(values, units=Units.KM)
    x = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, y.arctan2, x)

    values = np.random.randn(10)
    y = Scalar(values, units=Units.KM)
    x = Scalar(values, units=Units.UNITLESS)
    self.assertRaises(ValueError, y.arctan2, x)

    #-----------------------------
    # Units should be removed	      
    #-----------------------------
    values = np.random.randn(10)
    y = Scalar(values, units=Units.KM)
    x = Scalar(values, units=Units.CM)
    self.assertTrue(y.arctan2(x).units is None)

    #----------------------------
    # Units should be removed	     
    #----------------------------
    N = 100
    y = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    z = y.arctan2(x)
    self.assertTrue(np.all(z.mask[x.mask]))
    self.assertTrue(np.all(z.mask[y.mask]))
    self.assertTrue(not np.any(z.mask[~x.mask & ~y.mask]))

    #-------------------
    # Derivatives	    
    #-------------------
    N = 20
    y = Scalar(np.random.randn(N))
    x = Scalar(np.random.randn(N))
    x.insert_deriv('f', Scalar(np.random.randn(N)))
    x.insert_deriv('h', Scalar(np.random.randn(N)))
    y.insert_deriv('g', Scalar(np.random.randn(N)))
    y.insert_deriv('h', Scalar(np.random.randn(N)))

    self.assertIn('f', x.derivs)
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertNotIn('g', x.derivs)
    self.assertFalse(hasattr(x, 'd_dg'))
    self.assertIn('h', x.derivs)
    self.assertTrue(hasattr(x, 'd_dh'))

    self.assertNotIn('f', y.derivs)
    self.assertFalse(hasattr(y, 'd_df'))
    self.assertIn('g', y.derivs)
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertIn('h', y.derivs)
    self.assertTrue(hasattr(y, 'd_dh'))

    self.assertIn('f', y.arctan2(x).derivs)
    self.assertTrue(hasattr(y.arctan2(x), 'd_df'))
    self.assertIn('g', y.arctan2(x).derivs)
    self.assertTrue(hasattr(y.arctan2(x), 'd_dg'))
    self.assertIn('h', y.arctan2(x).derivs)
    self.assertTrue(hasattr(y.arctan2(x), 'd_dh'))

    EPS = 1.e-6
    z1 = y.arctan2(x + EPS)
    z0 = y.arctan2(x - EPS)
    dz_dx = 0.5 * (z1 - z0) / EPS

    z1 = (y + EPS).arctan2(x)
    z0 = (y - EPS).arctan2(x)
    dz_dy = 0.5 * (z1 - z0) / EPS

    z = y.arctan2(x)
    dz_df = z.d_df
    dz_dg = z.d_dg
    dz_dh = z.d_dh

    for i in range(N):
        self.assertAlmostEqual(dz_dx[i]*x.d_df[i], z.d_df[i], delta=EPS)
        self.assertAlmostEqual(dz_dy[i]*y.d_dg[i], z.d_dg[i], delta=EPS)
        self.assertAlmostEqual(dz_dx[i]*x.d_dh[i] + dz_dy[i]*y.d_dh[i],
                               z.d_dh[i], delta=EPS)

    #-----------------------------------------------
    # Derivatives should be removed if necessary    	
    #-----------------------------------------------
    self.assertEqual(y.arctan2(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.arctan2(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.arctan2(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.arctan2(x, recursive=False), 'd_dh'))

    #------------------------------------------
    # Read-only status should be preserved         
    #------------------------------------------
    N = 10
    y = Scalar(np.random.randn(N))
    x = Scalar(np.random.randn(N))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.arctan2(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().arctan2(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().arctan2(x).readonly)
    self.assertFalse(y.arctan2(x.as_readonly()).readonly)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
