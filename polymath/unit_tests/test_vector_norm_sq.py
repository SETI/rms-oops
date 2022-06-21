################################################################################
# Vector.norm_sq() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

#*******************************************************************************
# Test_Vector_norm_sq
#*******************************************************************************
class Test_Vector_norm_sq(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-------------------
    # Single values	    
    #-------------------
    x = Vector((-1.,))
    self.assertAlmostEqual(x.norm_sq(), 1.)

    x = Vector((1.,-2.,4.))
    self.assertAlmostEqual(x.norm_sq(), (1+4+16), 1.e-15)

    x = Vector((1.,2.,4.,8.), mask=True)
    self.assertTrue(x.norm_sq().mask is True)

    #----------------------
    # Arrays and masks	       
    #----------------------
    x = Vector(np.random.randn(3,7))
    n = x.norm_sq()
    self.assertTrue(not np.any(n.mask))

    N = 100
    x = Vector(np.random.randn(N,7),
               mask=(np.random.randn(N) < -0.3))    # Mask out a fraction
    n = x.norm_sq()

    #------------------------------
    # Test the unmasked items	       
    #------------------------------
    nn = n[~n.mask]
    xx = x[~n.mask]
    for i in range(len(nn)):
        self.assertAlmostEqual(nn[i], np.sum(xx.values[i]**2), delta=1.e-14)
        self.assertEqual(nn[i].mask, xx[i].mask)

    #------------------------------
    # Derivatives, denom = ()
    #------------------------------
    N = 100
    x = Vector(np.random.randn(N,3))

    x.insert_deriv('t', Vector(np.random.randn(N,3)))
    x.insert_deriv('v', Vector(np.random.randn(N,3,3), drank=1,
                               mask = (np.random.randn(N) < -0.4)))

    self.assertIn('t', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertIn('v', x.derivs)
    self.assertTrue(hasattr(x, 'd_dv'))

    y = x.norm_sq(recursive=False)
    self.assertNotIn('t', y.derivs)
    self.assertFalse(hasattr(y, 'd_dt'))
    self.assertNotIn('v', y.derivs)
    self.assertFalse(hasattr(y, 'd_dv'))

    y = x.norm_sq()
    self.assertIn('t', y.derivs)
    self.assertTrue(hasattr(y, 'd_dt'))
    self.assertIn('v', y.derivs)
    self.assertTrue(hasattr(y, 'd_dv'))

    EPS = 1.e-6
    y1 = (x + (EPS,0,0)).norm_sq()
    y0 = (x - (EPS,0,0)).norm_sq()
    dy_dx0 = 0.5 * (y1 - y0) / EPS

    y1 = (x + (0,EPS,0)).norm_sq()
    y0 = (x - (0,EPS,0)).norm_sq()
    dy_dx1 = 0.5 * (y1 - y0) / EPS

    y1 = (x + (0,0,EPS)).norm_sq()
    y0 = (x - (0,0,EPS)).norm_sq()
    dy_dx2 = 0.5 * (y1 - y0) / EPS

    dy_dt = (dy_dx0 * x.d_dt.values[:,0] +
             dy_dx1 * x.d_dt.values[:,1] +
             dy_dx2 * x.d_dt.values[:,2])

    dy_dv0 = (dy_dx0 * x.d_dv.values[:,0,0] +
              dy_dx1 * x.d_dv.values[:,1,0] +
              dy_dx2 * x.d_dv.values[:,2,0])

    dy_dv1 = (dy_dx0 * x.d_dv.values[:,0,1] +
              dy_dx1 * x.d_dv.values[:,1,1] +
              dy_dx2 * x.d_dv.values[:,2,1])

    dy_dv2 = (dy_dx0 * x.d_dv.values[:,0,2] +
              dy_dx1 * x.d_dv.values[:,1,2] +
              dy_dx2 * x.d_dv.values[:,2,2])

    for i in range(N):
        self.assertAlmostEqual(y.d_dt.values[i], dy_dt.values[i], delta=EPS)
        self.assertAlmostEqual(y.d_dv.values[i,0], dy_dv0.values[i], delta=EPS)
        self.assertAlmostEqual(y.d_dv.values[i,1], dy_dv1.values[i], delta=EPS)
        self.assertAlmostEqual(y.d_dv.values[i,2], dy_dv2.values[i], delta=EPS)

    #-----------------------------------------------
    # Read-only status should NOT be preserved	    	
    #-----------------------------------------------
    N = 10
    y = Vector(np.random.randn(N,3))
    x = Vector(np.random.randn(N,3))

    self.assertFalse(x.readonly)
    self.assertFalse(x.norm_sq().readonly)
    self.assertFalse(x.as_readonly().norm_sq().readonly)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
