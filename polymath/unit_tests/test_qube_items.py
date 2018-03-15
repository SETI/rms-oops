################################################################################
# Tests for Qube numerator and denominator methods
#
#   transpose_numer(self, axis1=0, axis2=1, recursive=True)
#   reshape_numer(self, shape, classes=(), recursive=True)
#   flatten_numer(self, classes=(), recursive=True)
# 
#   transpose_denom(self, axis1=0, axis2=1)
#   reshape_denom(self, shape)
#   flatten_denom(self)
#
#   join_items(self, classes)
#   swap_items(self, classes)
#   chain(self, arg)
#
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_items(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # transpose_numer(self, axis1=0, axis2=1, recursive=True)
    ############################################################################

    a = Matrix(np.random.randn(5,4,3,2), drank=1)
    b = a.transpose_numer(0,1)
    self.assertEqual(b.shape, (5,))
    self.assertEqual(b.numer, (3,4))
    self.assertEqual(b.denom, (2,))

    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.values[:,:,1] == b.values[:,1]))
    self.assertTrue(np.all(a.values[:,:,2] == b.values[:,2]))

    a.values[1,3,2] = 42.
    self.assertTrue(np.all(b.values[1,2,3] == 42))

    ####
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)

    b = a.transpose_numer(0,1,recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    b = a.transpose_numer(0,1,recursive=True)
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.values[:,:,1] == b.d_dt.values[:,1]))
    self.assertTrue(np.all(a.d_dt.values[:,:,2] == b.d_dt.values[:,2]))

    a.d_dt.values[1,1,2] = 42.
    self.assertTrue(np.all(b.d_dt.values[1,2,1] == 42))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)
    self.assertEqual(a.d_dt.readonly, False)
    self.assertEqual(b.d_dt.readonly, False)

    ####
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    a.as_readonly()

    b = a.transpose_numer(0,1,recursive=True)

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)
    self.assertEqual(a.d_dt.readonly, True)
    self.assertEqual(b.d_dt.readonly, True)

    ############################################################################
    # reshape_numer(self, shape, classes=(), recursive=True)
    ############################################################################

    a = Matrix(np.random.randn(5,4,3,2), drank=1)
    b = a.reshape_numer((6,2))
    self.assertEqual(b.shape, (5,))
    self.assertEqual(b.numer, (6,2))
    self.assertEqual(b.denom, (2,))

    self.assertTrue(np.all(a.values[:,0,0] == b.values[:,0,0]))
    self.assertTrue(np.all(a.values[:,0,1] == b.values[:,0,1]))
    self.assertTrue(np.all(a.values[:,0,2] == b.values[:,1,0]))
    self.assertTrue(np.all(a.values[:,1,0] == b.values[:,1,1]))
    self.assertTrue(np.all(a.values[:,1,1] == b.values[:,2,0]))
    self.assertTrue(np.all(a.values[:,1,2] == b.values[:,2,1]))
    self.assertTrue(np.all(a.values[:,2,0] == b.values[:,3,0]))
    self.assertTrue(np.all(a.values[:,2,1] == b.values[:,3,1]))
    self.assertTrue(np.all(a.values[:,2,2] == b.values[:,4,0]))
    self.assertTrue(np.all(a.values[:,3,0] == b.values[:,4,1]))
    self.assertTrue(np.all(a.values[:,3,1] == b.values[:,5,0]))
    self.assertTrue(np.all(a.values[:,3,2] == b.values[:,5,1]))

    a.values[1,3,2] = 42.
    self.assertTrue(np.all(b.values[1,5,1] == 42))

    ####
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)

    b = a.reshape_numer((6,2),recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    b = a.reshape_numer((6,2),recursive=True)
    self.assertTrue(np.all(a.d_dt.values[:,0,0] == b.d_dt.values[:,0,0]))
    self.assertTrue(np.all(a.d_dt.values[:,0,1] == b.d_dt.values[:,0,1]))
    self.assertTrue(np.all(a.d_dt.values[:,0,2] == b.d_dt.values[:,1,0]))
    self.assertTrue(np.all(a.d_dt.values[:,1,0] == b.d_dt.values[:,1,1]))
    self.assertTrue(np.all(a.d_dt.values[:,1,1] == b.d_dt.values[:,2,0]))
    self.assertTrue(np.all(a.d_dt.values[:,1,2] == b.d_dt.values[:,2,1]))
    self.assertTrue(np.all(a.d_dt.values[:,2,0] == b.d_dt.values[:,3,0]))
    self.assertTrue(np.all(a.d_dt.values[:,2,1] == b.d_dt.values[:,3,1]))
    self.assertTrue(np.all(a.d_dt.values[:,2,2] == b.d_dt.values[:,4,0]))
    self.assertTrue(np.all(a.d_dt.values[:,3,0] == b.d_dt.values[:,4,1]))
    self.assertTrue(np.all(a.d_dt.values[:,3,1] == b.d_dt.values[:,5,0]))
    self.assertTrue(np.all(a.d_dt.values[:,3,2] == b.d_dt.values[:,5,1]))

    a.d_dt.values[1,3,2] = 42.
    self.assertTrue(np.all(b.d_dt.values[1,5,1] == 42))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)
    self.assertEqual(a.d_dt.readonly, False)
    self.assertEqual(b.d_dt.readonly, False)

    ####
    a = Matrix(np.random.randn(5,4,3)).as_readonly()
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)

    b = a.reshape_numer((6,2),recursive=True)

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)
    self.assertEqual(a.d_dt.readonly, True)
    self.assertEqual(b.d_dt.readonly, True)

    a.as_readonly()
    self.assertEqual(a.d_dt.readonly, True)
    self.assertEqual(b.d_dt.readonly, True)

    ############################################################################
    # flatten_numer(self, classes=(), recursive=True)
    ############################################################################

    a = Matrix(np.random.randn(5,4,3,2), drank=1)
    b = a.flatten_numer()
    self.assertEqual(b.shape, (5,))
    self.assertEqual(b.numer, (12,))
    self.assertEqual(b.denom, (2,))

    self.assertTrue(np.all(a.values[:,0,0] == b.values[:,0]))
    self.assertTrue(np.all(a.values[:,0,1] == b.values[:,1]))
    self.assertTrue(np.all(a.values[:,0,2] == b.values[:,2]))
    self.assertTrue(np.all(a.values[:,1,0] == b.values[:,3]))
    self.assertTrue(np.all(a.values[:,1,1] == b.values[:,4]))
    self.assertTrue(np.all(a.values[:,1,2] == b.values[:,5]))
    self.assertTrue(np.all(a.values[:,2,0] == b.values[:,6]))
    self.assertTrue(np.all(a.values[:,2,1] == b.values[:,7]))
    self.assertTrue(np.all(a.values[:,2,2] == b.values[:,8]))
    self.assertTrue(np.all(a.values[:,3,0] == b.values[:,9]))
    self.assertTrue(np.all(a.values[:,3,1] == b.values[:,10]))
    self.assertTrue(np.all(a.values[:,3,2] == b.values[:,11]))

    a.values[1,3,2] = 42.
    self.assertTrue(np.all(b.values[1,11] == 42))

    ####
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)

    b = a.flatten_numer(recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    b = a.flatten_numer(recursive=True)
    self.assertTrue(np.all(a.d_dt.values[:,0,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.values[:,0,1] == b.d_dt.values[:,1]))
    self.assertTrue(np.all(a.d_dt.values[:,0,2] == b.d_dt.values[:,2]))
    self.assertTrue(np.all(a.d_dt.values[:,1,0] == b.d_dt.values[:,3]))
    self.assertTrue(np.all(a.d_dt.values[:,1,1] == b.d_dt.values[:,4]))
    self.assertTrue(np.all(a.d_dt.values[:,1,2] == b.d_dt.values[:,5]))
    self.assertTrue(np.all(a.d_dt.values[:,2,0] == b.d_dt.values[:,6]))
    self.assertTrue(np.all(a.d_dt.values[:,2,1] == b.d_dt.values[:,7]))
    self.assertTrue(np.all(a.d_dt.values[:,2,2] == b.d_dt.values[:,8]))
    self.assertTrue(np.all(a.d_dt.values[:,3,0] == b.d_dt.values[:,9]))
    self.assertTrue(np.all(a.d_dt.values[:,3,1] == b.d_dt.values[:,10]))
    self.assertTrue(np.all(a.d_dt.values[:,3,2] == b.d_dt.values[:,11]))

    a.d_dt.values[1,3,2] = 42.
    self.assertTrue(np.all(b.d_dt.values[1,11] == 42))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)
    self.assertEqual(a.d_dt.readonly, False)
    self.assertEqual(b.d_dt.readonly, False)

    ####
    a = Matrix(np.random.randn(5,4,3)).as_readonly()
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)

    b = a.flatten_numer(recursive=True)

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)
    self.assertEqual(a.d_dt.readonly, True)
    self.assertEqual(b.d_dt.readonly, True)

    ############################################################################
    # transpose_denom(self, axis1=0, axis2=1)
    ############################################################################

    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.transpose_denom(0,1)
    self.assertEqual(b.shape, (5,))
    self.assertEqual(b.numer, (4,))
    self.assertEqual(b.denom, (2,3))

    self.assertTrue(np.all(a.values[...,0] == b.values[...,0,:]))
    self.assertTrue(np.all(a.values[...,1] == b.values[...,1,:]))

    a.values[...,2,1] = 42.
    self.assertTrue(np.all(b.values[...,1,2] == 42))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    ####
    a = Matrix(np.random.randn(5,4,3,2), drank=2).as_readonly()
    b = a.transpose_denom(0,1)

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)

    ############################################################################
    # reshape_denom(self, shape)
    ############################################################################

    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.reshape_denom((2,3))
    self.assertEqual(b.shape, (5,))
    self.assertEqual(b.numer, (4,))
    self.assertEqual(b.denom, (2,3))

    self.assertTrue(np.all(a.values[...,0,0] == b.values[...,0,0]))
    self.assertTrue(np.all(a.values[...,0,1] == b.values[...,0,1]))
    self.assertTrue(np.all(a.values[...,1,0] == b.values[...,0,2]))
    self.assertTrue(np.all(a.values[...,1,1] == b.values[...,1,0]))
    self.assertTrue(np.all(a.values[...,2,0] == b.values[...,1,1]))
    self.assertTrue(np.all(a.values[...,2,1] == b.values[...,1,2]))

    a.values[1,1,2,1] = 42.
    self.assertTrue(np.all(b.values[1,1,1,2] == 42))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    ####
    a = Vector(np.random.randn(5,4,3,2), drank=2).as_readonly()
    b = a.reshape_denom((2,3))

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)

    ############################################################################
    # flatten_denom(self)
    ############################################################################

    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.flatten_denom()
    self.assertEqual(b.shape, (5,))
    self.assertEqual(b.numer, (4,))
    self.assertEqual(b.denom, (6,))

    self.assertTrue(np.all(a.values[...,0,0] == b.values[...,0]))
    self.assertTrue(np.all(a.values[...,0,1] == b.values[...,1]))
    self.assertTrue(np.all(a.values[...,1,0] == b.values[...,2]))
    self.assertTrue(np.all(a.values[...,1,1] == b.values[...,3]))
    self.assertTrue(np.all(a.values[...,2,0] == b.values[...,4]))
    self.assertTrue(np.all(a.values[...,2,1] == b.values[...,5]))

    a.values[1,1,2,1] = 42.
    self.assertTrue(np.all(b.values[1,1,5] == 42))

    ####
    a = Matrix(np.random.randn(5,4,3)).as_readonly()
    b = a.flatten_denom()

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)

    ############################################################################
    # join_items(self, classes)
    ############################################################################

    a = Vector(np.random.randn(5,4,3,2), drank=1)
    b = a.join_items(Matrix)

    self.assertEqual(b.shape, (5,4))
    self.assertEqual(b.numer, (3,2))
    self.assertEqual(b.denom, ())

    b = a.join_items((Boolean,Scalar,Matrix3,Quaternion,Matrix))
    self.assertEqual(type(b), Matrix)

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    a = a.as_readonly()
    b = a.join_items(Matrix)

    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)

    ############################################################################
    # swap_items(self, classes)
    ############################################################################

    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.swap_items((Boolean,Scalar,Matrix3,Quaternion,Matrix))
    self.assertEqual(type(b), Matrix)

    self.assertEqual(b.shape, a.shape)
    self.assertEqual(b.numer, a.denom)
    self.assertEqual(b.denom, a.numer)

    self.assertTrue(np.all(a.values[:,0] == b.values[...,0]))
    self.assertTrue(np.all(a.values[:,1] == b.values[...,1]))
    self.assertTrue(np.all(a.values[:,2] == b.values[...,2]))
    self.assertTrue(np.all(a.values[:,3] == b.values[...,3]))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    a = a.as_readonly()
    b = a.swap_items(Matrix)
    self.assertEqual(a.readonly, True)
    self.assertEqual(b.readonly, True)

    ############################################################################
    # chain(self, arg)
    ############################################################################

    a = Vector(np.arange(120).reshape((5,4,3,2)), drank=1)
    b = Vector(np.arange(60,180).reshape((5,4,2,3)), drank=1)

    a_values = a.values.reshape(5,4,3,2,1)
    b_values = b.values.reshape(5,4,1,2,3)
    a_chain_b_vals = np.sum(a_values * b_values, axis=-2)

    self.assertTrue(np.all(a.chain(b).values == a_chain_b_vals))
    self.assertEqual(a.chain(b).shape, (5,4))
    self.assertEqual(a.chain(b).numer, (3,))
    self.assertEqual(a.chain(b).denom, (3,))

    ####
    a = Vector(np.arange(60).reshape((5,3,4)), drank=1)
    b = Vector(np.arange(120).reshape((5,4,3,2)), drank=2)
    a_values = a.values.reshape(5,3,4,1,1)
    b_values = b.values.reshape(5,1,4,3,2)
    a_chain_b_vals = np.sum(a_values * b_values, axis=2)

    self.assertTrue(np.all(a.chain(b).values == a_chain_b_vals))
    self.assertEqual(a.chain(b).shape, (5,))
    self.assertEqual(a.chain(b).numer, (3,))
    self.assertEqual(a.chain(b).denom, (3,2))

    ####
    a = Vector(np.arange(120).reshape((5,4,3,2)), drank=2)
    b = Matrix(np.arange(270).reshape((5,3,2,3,3)), drank=2)
    a_values = a.values.reshape(5,4,6,1,1)
    b_values = b.values.reshape(5,1,6,3,3)
    a_chain_b_vals = np.sum(a_values * b_values, axis=2)

    self.assertTrue(np.all(a.chain(b).values == a_chain_b_vals))
    self.assertEqual(a_chain_b_vals.shape, (5,4,3,3))
    self.assertEqual(a.chain(b).shape, (5,))
    self.assertEqual(a.chain(b).numer, (4,))
    self.assertEqual(a.chain(b).denom, (3,3))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
