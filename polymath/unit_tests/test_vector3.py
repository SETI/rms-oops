################################################################################
# Vector3 tests for inherited methods
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Vector, Vector3, Matrix, Units

class Test_Vector3(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # arrays of wrong shape raise ValueError
        self.assertRaises(ValueError, Vector3, np.random.randn(3,4,5))
        self.assertRaises(ValueError, Vector3, 1.)

        # automatic coercion of booleans
        self.assertEqual(Vector3([True,True,False]), (1.,1.,0.))

        # zeros
        a = Vector3.zeros((2,3), dtype='int')
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.dtype.kind, 'f')    # coerced to float
        self.assertTrue(np.all(a.vals == 0))

        a = Vector3.zeros((2,3), dtype='float')
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.shape, (2,3,3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 0))

        a = Vector3.zeros((2,3), dtype='bool')
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.shape, (2,3,3))
        self.assertEqual(a.vals.dtype.kind, 'f')    # coerced to float
        self.assertTrue(np.all(a.vals == 0))

        a = Vector3.zeros((2,2), mask=[[0,1],[0,0]])
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3))
        self.assertTrue(np.all(a.vals == 0))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.mask == [[0,1],[0,0]]))

        a = Vector3.zeros((2,2), denom=(3,3))
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3,3,3))
        self.assertTrue(np.all(a.vals == 0))
        self.assertEqual(a.vals.dtype.kind, 'f')

        self.assertRaises(ValueError, Vector3.zeros, (2,3), numer=(4,))

        # ones
        a = Vector3.ones((2,3), dtype='int')
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 1))

        a = Vector3.ones((2,3), dtype='float')
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.shape, (2,3,3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 1))

        a = Vector3.ones((2,3), dtype='bool')
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.shape, (2,3,3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 1))

        a = Vector3.ones((2,2), mask=[[0,1],[0,0]])
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3))
        self.assertTrue(np.all(a.vals == 1))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.mask == [[0,1],[0,0]]))

        a = Vector3.ones((2,2), denom=(3,3))
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3,3,3))
        self.assertTrue(np.all(a.vals == 1))
        self.assertEqual(a.vals.dtype.kind, 'f')

        self.assertRaises(ValueError, Vector3.zeros, (2,3), numer=(4,))

        # filled
        a = Vector3.filled((2,3), 7)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 7))

        a = Vector3.filled((2,3), 7.)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.vals.shape, (2,3,3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 7))

        a = Vector3.filled((2,2), 7., mask=[[0,1],[0,0]])
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3))
        self.assertTrue(np.all(a.vals == 7))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.mask == [[0,1],[0,0]]))

        a = Vector3.filled((2,2), 7., denom=(3,3))
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3,3,3))
        self.assertTrue(np.all(a.vals == 7))
        self.assertEqual(a.vals.dtype.kind, 'f')

        a = Vector3.filled((2,2), (1.,2.,3.))
        self.assertEqual(a.shape, (2,2))
        self.assertEqual(a.vals.shape, (2,2,3))
        self.assertTrue(np.all(a.vals[...,0] == 1))
        self.assertTrue(np.all(a.vals[...,1] == 2))
        self.assertTrue(np.all(a.vals[...,2] == 3))
        self.assertEqual(a.vals.dtype.kind, 'f')

        self.assertRaises(ValueError, Vector3.zeros, (2,3), numer=(4,))

        # Most operations are inherited from Vector. These include:
        #     def to_scalar(self, axis, recursive=True)
        #     def to_scalars(self, recursive=True)
        #     def as_column(self, recursive=True)
        #     def as_row(self, recursive=True)
        #     def as_diagonal(self, recursive=True)
        #     def dot(self, arg, recursive=True)
        #     def norm(self, recursive=True)
        #     def unit(self, recursive=True)
        #     def cross(self, arg, recursive=True)
        #     def ucross(self, arg, recursive=True)
        #     def outer(self, arg, recursive=True)
        #     def perp(self, arg, recursive=True)
        #     def proj(self, arg, recursive=True)
        #     def sep(self, arg, recursive=True)
        #     def cross_product_as_matrix(self, recursive=True)
        #     def element_mul(self, arg, recursive=True):
        #     def element_div(self, arg, recursive=True):
        #     def __abs__(self)

        # Make sure proper objects are returned...
        a = Vector3(np.random.randn(4,1,5,3))
        b = Vector3(np.random.randn(8,5,3))

        self.assertEqual(type(a.to_scalar(0)), Scalar)
        self.assertEqual(a.to_scalar(0).shape, a.shape)

        self.assertEqual(len(a.to_scalars()), 3)
        self.assertEqual(type(a.to_scalars()[0]), Scalar)

        self.assertEqual(type(a.as_column()), Matrix)
        self.assertEqual(a.as_column().numer, (3,1))

        self.assertEqual(type(a.as_row()), Matrix)
        self.assertEqual(a.as_row().numer, (1,3))

        self.assertEqual(type(a.as_diagonal()), Matrix)
        self.assertEqual(a.as_diagonal().numer, (3,3))

        self.assertEqual(type(a.dot(b)), Scalar)
        self.assertEqual(type(a.norm()), Scalar)
        self.assertEqual(type(a.unit()), Vector3)
        self.assertEqual(type(a.cross(b)), Vector3)
        self.assertEqual(type(a.ucross(b)), Vector3)
        self.assertEqual(type(a.perp(b)), Vector3)
        self.assertEqual(type(a.proj(b)), Vector3)
        self.assertEqual(type(a.sep(b)), Scalar)

        self.assertEqual(type(a.cross_product_as_matrix()), Matrix)
        self.assertEqual(a.cross_product_as_matrix().numer, (3,3))

        self.assertEqual(type(a.element_mul(b)), Vector3)
        self.assertEqual(type(a.element_div(b)), Vector3)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
