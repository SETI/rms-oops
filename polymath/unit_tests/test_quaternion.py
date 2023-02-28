################################################################################
# Tests for Quaternion arithmetic and other operations
#     def as_quaternion(arg)
#     def from_rotation(angle, vector, recursive=True)
#     def conj(self, recursive=True)
#     def identity(self)
#     def reciprocal(self, recursive=True)
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Quaternion, Matrix3, Scalar

class Test_Quaternion(unittest.TestCase):

  # runTest
  def runTest(self):

    np.random.seed(8615)

    ############################################################################
    # as_quaternion(arg)
    ############################################################################

    a = Quaternion(np.random.randn(4))
    b = Quaternion.as_quaternion(a)
    self.assertTrue(a is b)

    a = Quaternion(np.random.randn(10,4))
    b = Quaternion.as_quaternion(a)
    self.assertTrue(a is b)

    a = (1,0,0,0)
    self.assertEqual(Quaternion.as_quaternion(a), a)

    a = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    self.assertEqual(Quaternion.as_quaternion(a), a)

    m = Matrix3((Matrix3.IDENTITY + 0.1 * np.random.randn(3,3)).unitary())
    q = Quaternion.as_quaternion(m)
    m2 = q.to_matrix3()

    DEL = 1.e-6
    self.assertTrue((m2 - m).rms() < DEL)

    N = 100
    m = Matrix3(N * [Matrix3.IDENTITY.values])
    m += 0.1 * np.random.randn(N,3,3)

    m = Matrix3(m.unitary())
    q = Quaternion.as_quaternion(m)
    m2 = q.to_matrix3()

    self.assertTrue((m2 - m).rms().max() < DEL)

    ############################################################################
    # from_rotation(angle, vector, recursive=True)
    ############################################################################

    a = Quaternion.from_rotation(np.pi/2., [(1,0,0),(0,1,0),(0,0,1)])

    DEL = 1.e-14
    self.assertAlmostEqual(a[0].values[0], np.sqrt(0.5), delta=DEL)
    self.assertAlmostEqual(a[0].values[1], np.sqrt(0.5), delta=DEL)
    self.assertAlmostEqual(a[0].values[2], 0., delta=DEL)
    self.assertAlmostEqual(a[0].values[3], 0., delta=DEL)

    self.assertAlmostEqual(a[1].values[0], np.sqrt(0.5), delta=DEL)
    self.assertAlmostEqual(a[1].values[1], 0., delta=DEL)
    self.assertAlmostEqual(a[1].values[2], np.sqrt(0.5), delta=DEL)
    self.assertAlmostEqual(a[1].values[3], 0., delta=DEL)

    self.assertAlmostEqual(a[2].values[0], np.sqrt(0.5), delta=DEL)
    self.assertAlmostEqual(a[2].values[1], 0., delta=DEL)
    self.assertAlmostEqual(a[2].values[2], 0., delta=DEL)
    self.assertAlmostEqual(a[2].values[3], np.sqrt(0.5), delta=DEL)

    angle = Scalar(0., derivs={'t': Scalar(1.)})
    a = Quaternion.from_rotation(angle, [(1,0,0),(0,1,0),(0,0,1)])
    self.assertEqual(a, (1,0,0,0))

    self.assertAlmostEqual(a.d_dt[0].values[0], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[0].values[1], 0.5, delta=DEL)
    self.assertAlmostEqual(a.d_dt[0].values[2], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[0].values[3], 0.0, delta=DEL)

    self.assertAlmostEqual(a.d_dt[1].values[0], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[1].values[1], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[1].values[2], 0.5, delta=DEL)
    self.assertAlmostEqual(a.d_dt[1].values[3], 0.0, delta=DEL)

    self.assertAlmostEqual(a.d_dt[2].values[0], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[2].values[1], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[2].values[2], 0.0, delta=DEL)
    self.assertAlmostEqual(a.d_dt[2].values[3], 0.5, delta=DEL)

    self.assertFalse(a.readonly)

    ############################################################################
    # conj(self, recursive=True)
    ############################################################################

    N = 100
    a = Quaternion(np.random.randn(N,4))
    a.insert_deriv('t', Quaternion(np.random.randn(N,4,2), drank=1))

    b = a.conj()
    (s,v) = b.to_parts()
    self.assertEqual(a.to_parts()[0],  b.to_parts()[0])
    self.assertEqual(a.to_parts()[1], -b.to_parts()[1])

    self.assertEqual(a.to_parts()[0].d_dt,  b.to_parts()[0].d_dt)
    self.assertEqual(a.to_parts()[1].d_dt, -b.to_parts()[1].d_dt)

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = a.as_readonly()
    b = a.conj()

    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    ############################################################################
    # def identity(self)
    ############################################################################

    b = a.identity()
    self.assertEqual(b, (1,0,0,0))

    ############################################################################
    # def reciprocal(self, recursive=True)
    ############################################################################

    a = Quaternion((1,0,0,0))
    self.assertEqual(a, a.reciprocal())
    self.assertFalse(a.reciprocal().readonly)

    N = 100
    a = Quaternion(np.random.randn(N,4),
                   derivs = {'t': Quaternion(np.random.randn(N,4,2), drank=1)})

    b = a.reciprocal()
    ab = a * b
    ba = b * a

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    DEL = 1.e-13
    for i in range(N):
        self.assertAlmostEqual(ab[i].values[0], 1., delta=DEL)
        self.assertAlmostEqual(ab[i].values[1], 0., delta=DEL)
        self.assertAlmostEqual(ab[i].values[2], 0., delta=DEL)
        self.assertAlmostEqual(ab[i].values[3], 0., delta=DEL)

        self.assertAlmostEqual(ba[i].values[0], 1., delta=DEL)
        self.assertAlmostEqual(ba[i].values[1], 0., delta=DEL)
        self.assertAlmostEqual(ba[i].values[2], 0., delta=DEL)
        self.assertAlmostEqual(ba[i].values[3], 0., delta=DEL)

    a = a.as_readonly()
    b = a.reciprocal()
    ab = a * b
    ba = b * a

    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(ab.readonly)
    self.assertFalse(ba.readonly)

    ############################################################################
    # Many operations are inherited from Vector. These include:
    #     def to_scalar(self, axis, recursive=True)
    #     def to_scalars(self, recursive=True)
    #     def norm(self, recursive=True)
    #     def norm_sq(self, recursive=True)
    #     def unit(self, recursive=True)
    #     def perp(self, arg, recursive=True)
    #     def proj(self, arg, recursive=True)
    #     def __abs__(self)
    #
    # Make sure these return the proper class...
    ############################################################################

    a = Quaternion([(1,0,0,0),(0,1,0,0)])

    self.assertEqual(type(a.to_scalar(0)), Scalar)

    self.assertEqual(len(a.to_scalars()), 4)
    self.assertEqual(type(a.to_scalars()), tuple)
    self.assertEqual(type(a.to_scalars()[0]), Scalar)

    self.assertEqual(type(a.norm()), Scalar)

    self.assertEqual(type(a.norm_sq()), Scalar)

    self.assertEqual(type(a.unit()), Quaternion)

    self.assertEqual(type(a.perp(a)), Quaternion)

    self.assertEqual(type(a.proj(a)), Quaternion)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
