################################################################################
# Vector.as_vector() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Vector, Scalar, Pair, Units

class Test_Vector_as_vector(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(4469)

    N = 10
    a = Vector(np.random.randn(N,6))
    da_dt = Vector(np.random.randn(N,6))
    a.insert_deriv('t', da_dt)

    b = Vector.as_vector(a, recursive=False)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))

    # Matrix case, Nx1
    a = Matrix(np.random.randn(N,7,1), units=Units.REV)
    da_dt = Matrix(np.random.randn(N,7,1,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Vector.as_vector(a)
    self.assertTrue(type(b), Vector)
    self.assertEqual(a.units, b.units)
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.numer, (7,1))
    self.assertEqual(b.numer, (7,))
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))

    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt.shape, b.shape)
    self.assertEqual(b.d_dt.numer, (7,))
    self.assertEqual(b.d_dt.denom, (6,))
    self.assertTrue(np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel()))

    b = Vector.as_vector(a, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    # Matrix case, 1xN
    a = Matrix(np.random.randn(N,1,7), units=Units.REV)
    da_dt = Matrix(np.random.randn(N,1,7,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Vector.as_vector(a)
    self.assertTrue(type(b), Vector)
    self.assertEqual(a.units, b.units)
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.numer, (1,7))
    self.assertEqual(b.numer, (7,))
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))

    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt.shape, b.shape)
    self.assertEqual(b.d_dt.numer, (7,))
    self.assertEqual(b.d_dt.denom, (6,))
    self.assertTrue(np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel()))

    b = Vector.as_vector(a, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    # Scalar case
    a = Scalar(np.random.randn(N), units=Units.UNITLESS)
    da_dt = Scalar(np.random.randn(N,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Vector.as_vector(a)
    self.assertTrue(type(b), Vector)
    self.assertEqual(a.units, b.units)
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.numer, ())
    self.assertEqual(b.numer, (1,))
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))

    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt.shape, b.shape)
    self.assertEqual(b.d_dt.numer, (1,))
    self.assertTrue(np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel()))

    b = Vector.as_vector(a, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    # Pair case
    a = Pair(np.random.randn(N,2), units=Units.DEG)
    da_dt = Pair(np.random.randn(N,2,6), drank=1)
    a.insert_deriv('t', da_dt)

    b = Vector.as_vector(a)
    self.assertTrue(type(b), Vector)
    self.assertEqual(a.units, b.units)
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.numer, b.numer)
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))

    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt.shape, b.shape)
    self.assertEqual(b.d_dt.numer, a.numer)
    self.assertTrue(np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel()))

    b = Vector.as_vector(a, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    # Other cases
    b = Vector.as_vector((1,2,3))
    self.assertTrue(type(b), Vector)
    self.assertTrue(b.units is None)
    self.assertEqual(b.shape, ())
    self.assertEqual(b.numer, (3,))
    self.assertEqual(b, (1,2,3))

    a = np.arange(120).reshape((2,4,3,5))
    b = Vector.as_vector(a)
    self.assertTrue(type(b), Vector)
    self.assertTrue(b.units is None)
    self.assertEqual(b.shape, (2,4,3))
    self.assertEqual(b.numer, (5,))
    self.assertEqual(b, a)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
