################################################################################
# Scalar.exp() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

#*******************************************************************************
# Test_Scalar_exp
#*******************************************************************************
class Test_Scalar_exp(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    # Individual values
    self.assertEqual(Scalar(1.25).exp(), np.exp(1.25))
    self.assertEqual(type(Scalar(1.25).exp()), Scalar)

    self.assertEqual(Scalar(1).exp(), np.exp(1.))
    self.assertEqual(Scalar(0).exp(), 1.)

    #----------------------
    # Multiple values       
    #----------------------
    self.assertEqual(Scalar((-1,0,1)).exp(), np.exp((-1,0,1)))
    self.assertEqual(type(Scalar((-1,0,1)).exp()), Scalar)

    #----------------
    # Arrays      
    #----------------
    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    funcvals = angles.exp()
    for i in range(N):
        self.assertEqual(funcvals[i], np.exp(values[i]))

    for i in range(N-1):
        self.assertEqual(funcvals[i:i+2], np.exp(values[i:i+2]))

    #----------------------
    # Test valid units       
    #----------------------
    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.KM)
    self.assertRaises(ValueError, Scalar.exp, random)

    values = np.random.randn(10) * 10.
    random = Scalar(values, units=Units.SECONDS)
    self.assertRaises(ValueError, Scalar.exp, random)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.exp(), np.exp(values))

    values = np.random.randn(10)
    random = Scalar(values, units=Units.UNITLESS)
    self.assertEqual(random.exp(), np.exp(values))

    #-----------------------------
    # Units should be removed      
    #-----------------------------
    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertTrue(random.exp().units is None)

    #--------------
    # Masks       
    #--------------
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.exp()
    self.assertTrue(np.all(y.mask[x.mask]))
    self.assertTrue(not np.any(y.mask[~x.mask]))

    #-----------------
    # Derivatives         
    #-----------------
    N = 100
    x = Scalar(np.random.randn(N) * 10.)
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))

    self.assertIn('t', x.derivs)
    self.assertIn('vec', x.derivs)
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))

    self.assertIn('t', x.exp().derivs)
    self.assertIn('vec', x.exp().derivs)
    self.assertTrue(hasattr(x.exp(), 'd_dt'))
    self.assertTrue(hasattr(x.exp(), 'd_dvec'))

    EPS = 1.e-6
    y1 = (x + EPS).exp()
    y0 = (x - EPS).exp()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.exp().d_dt
    dy_dvec = x.exp().d_dvec

    for i in range(N):
        self.assertAlmostEqual(dy_dx[i] * x.d_dt[i], dy_dt[i],
                               delta = max(1,abs(dy_dt[i])) * EPS)

        for k in range(3):
            self.assertAlmostEqual(dy_dx[i] * x.d_dvec[i].values[k],
                                   dy_dvec[i].values[k],
                                   delta = max(1,abs(dy_dvec[i].values[k]))*EPS)

    #---------------------------------------------
    # Derivatives should be removed if necessary
    #---------------------------------------------
    self.assertEqual(x.exp(recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_dt'))
    self.assertTrue(hasattr(x, 'd_dvec'))
    self.assertFalse(hasattr(x.exp(recursive=False), 'd_dt'))
    self.assertFalse(hasattr(x.exp(recursive=False), 'd_dvec'))

    #---------------------------------------------
    # Read-only status should NOT be preserved
    #---------------------------------------------
    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    self.assertFalse(x.readonly)
    self.assertFalse(x.exp().readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertFalse(x.as_readonly().exp().readonly)

    ###### With and without checking
    N = 1000
    x = Scalar(np.random.randn(N) * 700.)
    self.assertRaises(ValueError, x.log, check=False)

    self.assertTrue(x.exp(check=True).max() < np.inf)
    self.assertTrue(x.exp(check=True).max() > 1.e200)
    self.assertEqual(type(x.exp(check=True).mask), np.ndarray)
    self.assertTrue(np.sum(x.exp(check=True).mask) > 0)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
