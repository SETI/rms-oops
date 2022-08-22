################################################################################
# Vector.to_scalar(), Vector.to_scalars(), Vector.from_scalars()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Vector, Scalar, Pair, Units

#*******************************************************************************
# Test_Vector_scalars
#*******************************************************************************
class Test_Vector_scalars(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    self.assertTrue(np.all(a.values.ravel() == b.values.ravel()))
    self.assertEqual(a.shape, b.shape)
    self.assertEqual(a.values.shape, (N,1))
    self.assertEqual(b.values.shape, (N,))
    self.assertEqual(type(b), Scalar)

    c = a.to_scalars()
    self.assertTrue(np.all(a.values.ravel() == c[0].values.ravel()))
    self.assertEqual(a.shape, c[0].shape)
    self.assertEqual(b, c[0])
    self.assertEqual(type(c[0]), Scalar)

    #---------------------------
    # check units and masks
    #---------------------------
    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5),
               units=Units.RAD)
    c = a.to_scalars()
    self.assertEqual(a.units, c[0].units)

    b = a.to_scalar(1)
    self.assertEqual(b, c[1])

    self.assertTrue(np.all(b.values == a.values[...,1]))
    self.assertTrue(np.all(b.mask == a.mask))

    b[0] = 22.
    self.assertEqual(a[0].values[1], 22.)

    #------------------------
    # check derivatives
    #------------------------
    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    da_dt = Vector(np.random.randn(N,4))
    da_dv = Vector(np.random.randn(N,4,2), drank=1)

    a.insert_deriv('t', da_dt)
    a.insert_deriv('v', da_dv)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(a, 'd_dv'))

    b = a.to_scalar(3, recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dv'))

    b = a.to_scalar(3, recursive=True)
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dv'))

    self.assertEqual(b.d_dt.shape, a.shape)
    self.assertEqual(b.d_dt.numer, ())
    self.assertEqual(b.d_dt.denom, ())

    self.assertEqual(b.d_dv.shape, a.shape)
    self.assertEqual(b.d_dv.numer, ())
    self.assertEqual(b.d_dv.denom, (2,))

    self.assertTrue(np.all(a.values[...,3] == b.values))
    self.assertTrue(np.all(a.mask == b.mask))
    self.assertTrue(np.all(a.d_dt.values[...,3] == b.d_dt.values))
    self.assertTrue(np.all(a.d_dv.values[...,3,:] == b.d_dv.values))

    c = a.to_scalars(recursive=False)[3]
    self.assertFalse(hasattr(c, 'd_dt'))
    self.assertFalse(hasattr(c, 'd_dv'))

    c = a.to_scalars(recursive=True)[3]
    self.assertTrue(hasattr(c, 'd_dt'))
    self.assertTrue(hasattr(c, 'd_dv'))

    self.assertEqual(c.d_dt.shape, a.shape)
    self.assertEqual(c.d_dt.numer, ())
    self.assertEqual(c.d_dt.denom, ())

    self.assertEqual(c.d_dv.shape, a.shape)
    self.assertEqual(c.d_dv.numer, ())
    self.assertEqual(c.d_dv.denom, (2,))

    self.assertTrue(np.all(a.values[...,3] == c.values))
    self.assertTrue(np.all(a.mask == c.mask))
    self.assertTrue(np.all(a.d_dt.values[...,3] == c.d_dt.values))
    self.assertTrue(np.all(a.d_dv.values[...,3,:] == c.d_dv.values))

    #-------------------------
    # read-only status
    #-------------------------
    N = 10
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    self.assertFalse(a.readonly)

    b = a.to_scalar(3)
    self.assertFalse(b.readonly)

    c = a.to_scalars()[3]
    self.assertFalse(c.readonly)

    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    a.as_readonly()
    self.assertTrue(a.readonly)

    b = a.to_scalar(3)
    self.assertTrue(b.readonly)     # because of memory overlap

    c = a.to_scalars()[3]
    self.assertTrue(c.readonly)     # because of memory overlap

    #-------------------------
    # from_scalars(*args)
    #-------------------------
    a = 1.
    b = Scalar((2,3,4), mask=(True,False,False))
    c = np.random.randn(4,3)

    test = Vector.from_scalars(a,b,c)

    self.assertTrue(np.all(test.values[...,0] == 1))
    self.assertTrue(np.all(test.values[...,1] == (2,3,4)))
    self.assertTrue(np.all(test.values[...,2] == c))
    self.assertTrue(np.all(test.mask == [True,False,False]))
    self.assertEqual(test.readonly, False)

    b = b.as_readonly()
    c = Scalar(c).as_readonly()
    test = Vector.from_scalars(a,b,c)
    self.assertEqual(test.readonly, False)

    #------------------------------------------
    # from_scalars(*args), with derivatives
    #------------------------------------------
    a = 1.
    b = Scalar([2,3,4], mask=(True,False,False))
    c = np.random.randn(4,3)

    b.insert_deriv('t', Scalar([3,4,5], mask=(False,True,False)))

    test = Vector.from_scalars(a,b,c, recursive=True)

    self.assertTrue(np.all(test.values[...,0] == 1))
    self.assertTrue(np.all(test.values[...,1] == (2,3,4)))
    self.assertTrue(np.all(test.values[...,2] == c))
    self.assertTrue(np.all(test.mask == [True,False,False]))

    self.assertEqual(test.readonly, False)

    self.assertEqual(test.d_dt.values.shape, (4,3,3))
    self.assertTrue(np.all(test.d_dt.values[...,0] == 0))
    self.assertTrue(np.all(test.d_dt.values[...,1] == (3,4,5)))
    self.assertTrue(np.all(test.d_dt.values[...,2] == 0))

    #---------------------------------------------------------
    # from_scalars(*args), with derivatives, denominators
    #---------------------------------------------------------
    a = 1.

    b = Scalar((2,3,4), mask=(True,False,False))    # shape=(3,), item=()
    db_dt = Scalar(np.arange(100,112).reshape(3,2,2), drank=2,
                   mask=[False,True,False])
    b.insert_deriv('t', db_dt)

    c = Scalar(np.random.randn(4,3), mask=(np.random.rand(4,3) < 0.3))
                                                    # shape=(4,3), item=()

    #-------------------------
    # c.mask is random 4x3
    #-------------------------
    dc_dt = Scalar(np.random.randn(4,3,2,2), drank=2, mask=c.mask)
    c.insert_deriv('t', dc_dt)

    abc = Vector.from_scalars(a, b, c, recursive=True) # shape=(4,3), item=(3,)

    #--------------------------------------------
    # abc inherits c.mask, or'ed with  b.mask
    #--------------------------------------------
    self.assertTrue(np.all(abc.values[...,0] == 1))
    self.assertTrue(np.all(abc.values[...,1] == (2,3,4)))
    self.assertTrue(np.all(abc.values[...,2] == c.values))
    self.assertTrue(np.all(abc.mask == (c.mask | [True,False,False])))

    self.assertEqual(abc.readonly, False)

    self.assertEqual(abc.d_dt.values.shape, (4,3,3,2,2))

    self.assertTrue(np.all(abc.d_dt.values[...,0,:,:] == 0))
    self.assertTrue(np.all(abc.d_dt.values[...,1,:,:].flatten() ==
                           4*list(range(100,112))))
    self.assertTrue(np.all(abc.d_dt.values[...,2,:,:] == c.d_dt.values))

    self.assertTrue(np.all(abc.d_dt.mask == (db_dt.mask | dc_dt.mask)))
  #=============================================================================



#*******************************************************************************


################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
