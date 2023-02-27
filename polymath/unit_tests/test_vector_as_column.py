################################################################################
# Vector.as_column() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Vector, Scalar, Pair, Units

class Test_Vector_as_column(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(1684)

    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.as_column()
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.values.shape, (N,1))
    self.assertEqual(b.values.shape, (N,1,1))
    self.assertEqual(type(b), Matrix)

    # check units and masks
    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5),
               units=Units.RAD)
    b = a.as_column()
    self.assertEqual(a.units, b.units)

    self.assertTrue(np.all(b.values[...,0] == a.values))
    self.assertTrue(np.all(b.mask == a.mask))

    a.values[0,0] = 22.
    self.assertEqual(b.values[0,0,0], 22.)

    # check derivatives
    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    da_dt = Vector(np.random.randn(N,4))
    da_dv = Vector(np.random.randn(N,4,2), drank=1)

    a.insert_deriv('t', da_dt)
    a.insert_deriv('v', da_dv)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(a, 'd_dv'))

    b = a.as_column(recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dv'))

    b = a.as_column(recursive=True)
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dv'))

    self.assertEqual(b.d_dt.shape, a.shape)
    self.assertEqual(b.d_dt.numer, (4,1))
    self.assertEqual(b.d_dt.denom, ())

    self.assertEqual(b.d_dv.shape, a.shape)
    self.assertEqual(b.d_dv.numer, (4,1))
    self.assertEqual(b.d_dv.denom, (2,))

    self.assertTrue(np.all(a.values == b.values[...,0]))
    self.assertTrue(np.all(a.mask == b.mask))
    self.assertTrue(np.all(a.d_dt.values == b.d_dt.values[...,0]))
    self.assertTrue(np.all(a.d_dv.values == b.d_dv.values[...,0,:]))

    # read-only status
    N = 10
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    self.assertFalse(a.readonly)

    b = a.as_column()
    self.assertFalse(b.readonly)

    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    a = a.as_readonly()
    self.assertTrue(a.readonly)

    b = a.as_column()
    self.assertTrue(b.readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
