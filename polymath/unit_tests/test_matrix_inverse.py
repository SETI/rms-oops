################################################################################
# Matrix.inverse() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Scalar, Units

class Test_Matrix_inverse(unittest.TestCase):

  def runTest(self):

    DEL = 6.e-12

    # Make sure 3x3 matrix inversion is successful
    a = np.random.randn(3,3)
    b = Matrix.inverse_3x3(a)[0]

    a = Matrix(a)
    b = Matrix(b)

    axb = a * b
    bxa = b * a
    for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(axb.values[j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[j,k], int(j==k), delta=DEL)

    N = 30
    mats = np.random.randn(N,3,3)
    invs = Matrix.inverse_3x3(mats)

    a = Matrix(mats)
    b = Matrix(invs[0], mask = invs[1])
    self.assertTrue(not np.any(b.mask))

    axb = a * b
    bxa = b * a

    for i in range(N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(axb.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[i,j,k], int(j==k), delta=DEL)

    # Make sure 2x2 matrix inversion is successful
    a = np.random.randn(2,2)
    b = Matrix.inverse_2x2(a)[0]

    a = Matrix(a)
    b = Matrix(b)

    axb = a * b
    bxa = b * a
    for j in range(2):
        for k in range(2):
            self.assertAlmostEqual(axb.values[j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[j,k], int(j==k), delta=DEL)

    N = 30
    mats = np.random.randn(N,2,2)
    invs = Matrix.inverse_2x2(mats)

    a = Matrix(mats)
    b = Matrix(invs[0], mask = invs[1])
    self.assertTrue(not np.any(b.mask))

    axb = a * b
    bxa = b * a

    for i in range(N):
      for j in range(2):
        for k in range(2):
            self.assertAlmostEqual(axb.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[i,j,k], int(j==k), delta=DEL)

    # Make sure diagonal matrix inversion is successful
    a = np.diag(np.random.randn(6))
    b = Matrix.inverse_diag(a)[0]

    a = Matrix(a)
    b = Matrix(b)

    axb = a * b
    bxa = b * a
    for j in range(6):
        for k in range(6):
            self.assertAlmostEqual(axb.values[j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[j,k], int(j==k), delta=DEL)

    N = 30
    size = 5
    mats = np.random.randn(N,size,size)
    for i in range(N):
      for j in range(size):
        for k in range(size):
            if j != k: mats[i,j,k] = 0.

    invs = Matrix.inverse_diag(mats)

    a = Matrix(mats)
    b = Matrix(invs[0], mask = invs[1])
    self.assertTrue(not np.any(b.mask))

    axb = a * b
    bxa = b * a

    for i in range(N):
      for j in range(size):
        for k in range(size):
            self.assertAlmostEqual(axb.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[i,j,k], int(j==k), delta=DEL)

    # Invert 3x3, with first matrix uninvertible
    N = 30
    values = np.random.randn(N,3,3)
    values[0,0,0] = 0.
    values[0,0,1] = 0.
    values[0,0,2] = 0.

    a = Matrix(values)
    b = a.inverse()
    axb = a * b
    bxa = b * a

    self.assertTrue(b.mask[0])
    self.assertFalse(np.any(b.mask[1:]))

    for i in range(1,N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(axb.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[i,j,k], int(j==k), delta=DEL)

    # Invert 2x2, with first matrix uninvertible
    N = 30
    values = np.random.randn(N,2,2)
    values[0,0,0] = 0.
    values[0,0,1] = 0.

    a = Matrix(values)
    b = a.inverse()
    axb = a * b
    bxa = b * a

    self.assertTrue(b.mask[0])
    self.assertFalse(np.any(b.mask[1:]))

    for i in range(1,N):
      for j in range(2):
        for k in range(2):
            self.assertAlmostEqual(axb.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[i,j,k], int(j==k), delta=DEL)

    # Invert diagonal, with first matrix uninvertible
    N = 30
    size = 5
    values = np.random.randn(N,size,size)
    for i in range(N):
      for j in range(size):
        for k in range(size):
            if j != k: values[i,j,k] = 0.

    values[0,0,0] = 0.

    a = Matrix(values)
    b = a.inverse()
    axb = a * b
    bxa = b * a

    self.assertTrue(b.mask[0])
    self.assertFalse(np.any(b.mask[1:]))

    for i in range(1,N):
      for j in range(2):
        for k in range(2):
            self.assertAlmostEqual(axb.values[i,j,k], int(j==k), delta=DEL)
            self.assertAlmostEqual(bxa.values[i,j,k], int(j==k), delta=DEL)

    # Anything else raises an error
    a = Matrix(np.random.randn(N,3,4))
    self.assertRaises(ValueError, a.inverse)

    a = Matrix(np.random.randn(N,4,4))
    self.assertRaises(NotImplementedError, a.inverse)

    a = Matrix(np.random.randn(N,3,3,2,4), drank=2)
    self.assertRaises(ValueError, a.inverse)

    # Test units
    N = 5
    a = Matrix(np.random.randn(N,3,3), units=Units.CM**2/Units.S)
    b = a.inverse()
    self.assertEquals(b.units, Units.S/Units.CM**2)

    # Derivatives, 3x3
    N = 30
    a = Matrix(np.random.randn(N,3,3))
    a.insert_deriv('t', Matrix(np.random.randn(N,3,3)))
    a.insert_deriv('v', Matrix(np.random.randn(N,3,3,2), drank=1))

    self.assertIn('t', a.derivs)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertIn('v', a.derivs)
    self.assertTrue(hasattr(a, 'd_dv'))

    b = a.inverse(recursive=False)

    self.assertNotIn('t', b.derivs)
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertNotIn('v', b.derivs)
    self.assertFalse(hasattr(b, 'd_dv'))

    b = a.inverse(recursive=True)

    self.assertIn('t', b.derivs)
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertIn('v', b.derivs)
    self.assertTrue(hasattr(b, 'd_dv'))

    EPS = 1.e-6

    db_da_values = np.empty((N,3,3,3,3))

    for i in range(3):
      for j in range(3):
        da = np.zeros((3,3))
        da[i,j] = EPS
        b1 = (a + da).inverse()
        b0 = (a - da).inverse()
        db_da_values[...,i,j] = (0.5/EPS) * (b1 - b0).values

    db_da = Matrix(db_da_values, drank=2)

    db_dt = db_da.chain(a.d_dt)
    db_dv = db_da.chain(a.d_dv)

    tscale = np.sqrt(np.mean(np.mean(db_dt.values**2, axis=-1), axis=-1))
    vscale = np.sqrt(np.mean(np.mean(db_dv.values**2, axis=-2), axis=-2))

    DEL = 1.e-4
    for i in range(N):
      for j in range(3):
        for k in range(3):
            self.assertAlmostEqual(db_dt.values[i,j,k],
                                   b.d_dt.values[i,j,k],
                                   delta = DEL * max(1., tscale[i]))
            self.assertAlmostEqual(db_dv.values[i,j,k,0],
                                   b.d_dv.values[i,j,k,0],
                                   delta = DEL * max(1., vscale[i,0]))
            self.assertAlmostEqual(db_dv.values[i,j,k,1],
                                   b.d_dv.values[i,j,k,1],
                                   delta = DEL * max(1., vscale[i,1]))

    # Read-only status should NOT be preserved
    N = 10
    a = Matrix(np.random.randn(N,3,3))
    b = a.inverse()

    self.assertFalse(a.readonly)
    self.assertFalse(a.inverse().readonly)
    self.assertTrue(a.as_readonly().readonly)
    self.assertFalse(a.as_readonly().inverse().readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
