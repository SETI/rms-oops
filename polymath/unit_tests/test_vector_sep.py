################################################################################
# Vector.sep() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Units

#*******************************************************************************
# Test_Vector_sep
#*******************************************************************************
class Test_Vector_sep(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #------------------
    # Single values
    #------------------
    DEL = 1.e-12
    a = Vector((2,0,0))
    self.assertAlmostEqual(a.sep(Vector((0,1,0))),  0.50 * np.pi, delta=DEL)
    self.assertAlmostEqual(a.sep(Vector((1,0,1))),  0.25 * np.pi, delta=DEL)
    self.assertAlmostEqual(a.sep(Vector((-1,0,1))), 0.75 * np.pi, delta=DEL)
    self.assertAlmostEqual(a.sep(Vector((-1,0,0))), 1.00 * np.pi, delta=DEL)

    #----------------------
    # Multiple values
    #----------------------
    N = 100
    a = Vector(np.random.randn(N,3))
    b = Vector(np.random.randn(N,3))
    sep = a.sep(b)

    sep1 = a.unit().dot(b.unit()).arccos()

    for i in range(N):
        self.assertAlmostEqual(sep[i], sep1[i], delta=1.e-10)

    sep2 = a.unit().cross(b.unit()).norm().arcsin()
    mask = (a.dot(b) < 0.)
    sep2[mask] = np.pi - sep2[mask]

    for i in range(N):
        self.assertAlmostEqual(sep[i], sep2[i], delta=2.e-10)

    #----------------
    # Test units
    #----------------
    N = 10
    a = Vector(np.random.randn(N,3), units=Units.KM)
    b = Vector(np.random.randn(N,3), units=Units.KM)
    self.assertTrue(a.sep(b).mask is False)
    self.assertTrue(a.sep(b).units is None)

    a = Vector(np.random.randn(N,3), units=Units.KM)
    b = Vector(np.random.randn(N,3), units=Units.CM)
    self.assertTrue(a.sep(b).mask is False)
    self.assertTrue(a.sep(b).units is None)

    a = Vector(np.random.randn(N,3), units=Units.KM)
    b = Vector(np.random.randn(N,3), units=Units.S)
    self.assertTrue(a.sep(b).mask is False)
    self.assertTrue(a.sep(b).units is None)

    #-------------------
    # Derivatives
    #-------------------
    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))

    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))

    z = y.sep(x)

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

    self.assertIn('f', z.derivs)
    self.assertTrue(hasattr(z, 'd_df'))
    self.assertIn('g', z.derivs)
    self.assertTrue(hasattr(z, 'd_dg'))
    self.assertIn('h', z.derivs)
    self.assertTrue(hasattr(z, 'd_dh'))

    EPS = 1.e-6
    z1 = y.sep(x + (EPS,0,0))
    z0 = y.sep(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS

    z1 = y.sep(x + (0,EPS,0))
    z0 = y.sep(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS

    z1 = y.sep(x + (0,0,EPS))
    z0 = y.sep(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (EPS,0,0)).sep(x)
    z0 = (y - (EPS,0,0)).sep(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,EPS,0)).sep(x)
    z0 = (y - (0,EPS,0)).sep(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS

    z1 = (y + (0,0,EPS)).sep(x)
    z0 = (y - (0,0,EPS)).sep(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS

    dz_df = (dz_dx0 * x.d_df.values[:,0] +
             dz_dx1 * x.d_df.values[:,1] +
             dz_dx2 * x.d_df.values[:,2])

    dz_dg = (dz_dy0 * y.d_dg.values[:,0] +
             dz_dy1 * y.d_dg.values[:,1] +
             dz_dy2 * y.d_dg.values[:,2])

    dz_dh = (dz_dx0 * x.d_dh.values[:,0] + dz_dy0 * y.d_dh.values[:,0] +
             dz_dx1 * x.d_dh.values[:,1] + dz_dy1 * y.d_dh.values[:,1] +
             dz_dx2 * x.d_dh.values[:,2] + dz_dy2 * y.d_dh.values[:,2])

    for i in range(N):
        self.assertAlmostEqual(z.d_df.values[i], dz_df.values[i], delta=EPS)
        self.assertAlmostEqual(z.d_dg.values[i], dz_dg.values[i], delta=EPS)
        self.assertAlmostEqual(z.d_dh.values[i], dz_dh.values[i], delta=EPS)

    #-----------------------------------------------
    # Derivatives should be removed if necessary
    #-----------------------------------------------
    self.assertEqual(y.sep(x, recursive=False).derivs, {})
    self.assertTrue(hasattr(x, 'd_df'))
    self.assertTrue(hasattr(x, 'd_dh'))
    self.assertTrue(hasattr(y, 'd_dg'))
    self.assertTrue(hasattr(y, 'd_dh'))
    self.assertFalse(hasattr(y.sep(x, recursive=False), 'd_df'))
    self.assertFalse(hasattr(y.sep(x, recursive=False), 'd_dg'))
    self.assertFalse(hasattr(y.sep(x, recursive=False), 'd_dh'))

    #-----------------------------------------------
    # Read-only status should NOT be preserved
    #-----------------------------------------------
    N = 10
    y = Vector(np.random.randn(N,7))
    x = Vector(np.random.randn(N,7))

    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertFalse(y.sep(x).readonly)

    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)
    self.assertFalse(y.as_readonly().sep(x.as_readonly()).readonly)

    self.assertFalse(y.as_readonly().sep(x).readonly)
    self.assertFalse(y.sep(x.as_readonly()).readonly)
  #=============================================================================



#*******************************************************************************


################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
