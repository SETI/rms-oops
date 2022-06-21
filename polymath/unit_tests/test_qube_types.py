################################################################################
# Tests for Qube methods related to data types
#   mvals(self)
#   is_numeric(self)
#   as_numeric(self)
#   is_float(self)
#   as_float(self)
#   is_int(self)
#   as_int(self)
#   masked_single(self)
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

#*******************************************************************************
# Test_Qube_types
#*******************************************************************************
class Test_Qube_types(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    ############################################################################
    # mvals(self)
    ############################################################################

    a = Scalar(np.random.randn(4,5), mask=(np.random.rand(4,5) < 0.2))

    mv = a.mvals
    self.assertTrue(np.all(mv.data == a.values))
    self.assertTrue(np.all(mv.mask == a.mask))

    a = Vector(np.random.randn(4,5,3), mask=(np.random.rand(4,5) < 0.2))
    mv = a.mvals
    self.assertTrue(np.all(mv.data == a.values))
    self.assertTrue(np.all(mv.mask[...,0] == a.mask))
    self.assertTrue(np.all(mv.mask[...,1] == a.mask))
    self.assertTrue(np.all(mv.mask[...,2] == a.mask))

    a = Matrix(np.random.randn(4,5,3,3), mask=(np.random.rand(4,5) < 0.2))
    mv = a.mvals
    self.assertTrue(np.all(mv.data == a.values))
    self.assertTrue(np.all(mv.mask == a.mask[...,np.newaxis,np.newaxis]))

    ############################################################################
    # is_numeric(self)
    ############################################################################

    self.assertEqual(Boolean.TRUE.is_numeric(), False)
    self.assertEqual(Scalar.ONE.is_numeric(), True)
    self.assertEqual(Vector.XAXIS3.is_numeric(), True)
    self.assertEqual(Vector3.XAXIS.is_numeric(), True)
    self.assertEqual(Pair.XAXIS.is_numeric(), True)
    self.assertEqual(Matrix.IDENTITY2.is_numeric(), True)
    self.assertEqual(Matrix3.IDENTITY.is_numeric(), True)

    ############################################################################
    # is_numeric(self)
    ############################################################################

    self.assertEqual(Boolean.TRUE.is_numeric(), False)
    self.assertEqual(Scalar.ONE.is_numeric(), True)
    self.assertEqual(Vector.XAXIS3.is_numeric(), True)
    self.assertEqual(Vector3.XAXIS.is_numeric(), True)
    self.assertEqual(Pair.XAXIS.is_numeric(), True)
    self.assertEqual(Matrix.IDENTITY2.is_numeric(), True)
    self.assertEqual(Matrix3.IDENTITY.is_numeric(), True)

    ############################################################################
    # as_numeric(self)
    ############################################################################

    self.assertEqual(Boolean.TRUE.as_numeric(), 1)
    self.assertEqual(Boolean.FALSE.as_numeric(), 0)
    self.assertEqual(type(Boolean.TRUE.as_numeric()), Scalar)
    self.assertEqual(type(Boolean.FALSE.as_numeric()), Scalar)

    self.assertEqual(Scalar.ONE.as_numeric(), Scalar.ONE)
    self.assertEqual(Vector.XAXIS3.as_numeric(), Vector.XAXIS3)
    self.assertEqual(Vector3.XAXIS.as_numeric(), Vector3.XAXIS)
    self.assertEqual(Pair.XAXIS.as_numeric(), Pair.XAXIS)
    self.assertEqual(Matrix.IDENTITY2.as_numeric(), Matrix.IDENTITY2)
    self.assertEqual(Matrix3.IDENTITY.as_numeric(), Matrix3.IDENTITY)

    ############################################################################
    # is_float(self)
    # is_int(self)
    ############################################################################

    self.assertEqual(Boolean((True,False)).is_int(), False)
    self.assertEqual(Boolean((True,False)).is_float(), False)

    self.assertEqual(Scalar((1,2,3)).is_int(), True)
    self.assertEqual(Scalar((1,2,3)).is_float(), False)
    self.assertEqual(Scalar((1.,2.,3.)).is_int(), False)
    self.assertEqual(Scalar((1.,2.,3.)).is_float(), True)

    self.assertEqual(Vector((1,2,3)).is_int(), True)
    self.assertEqual(Vector((1,2,3)).is_float(), False)
    self.assertEqual(Vector((1.,2.,3.)).is_int(), False)
    self.assertEqual(Vector((1.,2.,3.)).is_float(), True)

    self.assertEqual(Vector3((1,2,3)).is_int(), False)      # coerced to float
    self.assertEqual(Vector3((1,2,3)).is_float(), True)
    self.assertEqual(Vector3((1.,2.,3.)).is_int(), False)
    self.assertEqual(Vector3((1.,2.,3.)).is_float(), True)

    self.assertEqual(Pair((1,2)).is_int(), True)
    self.assertEqual(Pair((1,2)).is_float(), False)
    self.assertEqual(Pair((1.,2.)).is_int(), False)
    self.assertEqual(Pair((1.,2.)).is_float(), True)

    self.assertEqual(Quaternion((1,2,3,4)).is_int(), False) # coerced to float
    self.assertEqual(Quaternion((1,2,3,4)).is_float(), True)
    self.assertEqual(Quaternion((1.,2.,3.,4.)).is_int(), False)
    self.assertEqual(Quaternion((1.,2.,3.,4.)).is_float(), True)

    self.assertEqual(Matrix([(1,2),(3,4)]).is_int(), False) # coerced to float
    self.assertEqual(Matrix([(1,2),(3,4)]).is_float(), True)
    self.assertEqual(Matrix([(1.,2.),(3.,4.)]).is_int(), False)
    self.assertEqual(Matrix([(1.,2.),(3.,4.)]).is_float(), True)

    ############################################################################
    # as_float(self)
    # as_int(self)
    ############################################################################

    self.assertEqual(Boolean(True).as_int(), 1)
    self.assertEqual(Boolean(False).as_int(), 0)
    self.assertEqual(type(Boolean(True).as_int()), Scalar)
    self.assertEqual(type(Boolean(False).as_int()), Scalar)
    self.assertEqual(type(Boolean(True).as_int().values), int)
    self.assertEqual(type(Boolean(False).as_int().values), int)

    self.assertEqual(Boolean(True).as_float(), 1)
    self.assertEqual(Boolean(False).as_float(), 0)
    self.assertEqual(type(Boolean(True).as_float()), Scalar)
    self.assertEqual(type(Boolean(False).as_float()), Scalar)
    self.assertEqual(type(Boolean(True).as_float().values), float)
    self.assertEqual(type(Boolean(False).as_float().values), float)

    self.assertEqual(Boolean((True,False)).as_int(), (1,0))
    self.assertEqual(type(Boolean((True,False)).as_int()), Scalar)
    self.assertEqual(Boolean((True,False)).as_int().values.dtype,
                     np.dtype('int'))

    self.assertEqual(Boolean((True,False)).as_float(), (1,0))
    self.assertEqual(type(Boolean((True,False)).as_float()), Scalar)
    self.assertEqual(Boolean((True,False)).as_float().values.dtype,
                     np.dtype('float'))

    self.assertEqual(type(Scalar(1.).as_int().values), int)
    self.assertEqual(Scalar((1.,2.)).as_int().values.dtype, np.dtype('int'))
    self.assertEqual(Scalar((1.5,-1.5)).as_int(), (1,-2))

    self.assertEqual(type(Scalar(1).as_float().values), float)
    self.assertEqual(Scalar((1,2)).as_float().values.dtype, np.dtype('float'))

    self.assertEqual(Vector((1.,2.)).as_int().values.dtype, np.dtype('int'))
    self.assertEqual(Vector((1.5,-1.5)).as_int().values.dtype, np.dtype('int'))

    self.assertEqual(Vector((1,2)).as_float().values.dtype, np.dtype('float'))

    self.assertEqual(Pair((1.,2.)).as_int().values.dtype, np.dtype('int'))
    self.assertEqual(Pair((1.5,-1.5)).as_int().values.dtype, np.dtype('int'))

    self.assertEqual(Pair((1,2)).as_float().values.dtype, np.dtype('float'))

    self.assertRaises(TypeError, Vector3((1.,2.,3.)).as_int)
    self.assertRaises(TypeError, Quaternion((1.,2.,3.,4.)).as_int)
    self.assertRaises(TypeError, Matrix([(1,0),(0,1)]).as_int)
    self.assertRaises(TypeError, Matrix3([(1,0,0),(0,1,0),(0,0,1)]).as_int)

    ############################################################################
    # masked_single(self)
    ############################################################################

    a = Scalar((1,2,3))
    self.assertEqual(a.masked_single(), Scalar.MASKED)
    self.assertEqual(type(a.masked_single()), Scalar)
    self.assertEqual(a.masked_single().shape, ())

    a = Boolean([True,False])
    self.assertEqual(a.masked_single(), Boolean.MASKED)
    self.assertEqual(type(a.masked_single()), Boolean)
    self.assertEqual(a.masked_single().shape, ())

    a = Vector([(1,2,3),(4,5,6)])
    self.assertEqual(a.masked_single(), Vector.MASKED3)
    self.assertEqual(type(a.masked_single()), Vector)
    self.assertEqual(a.masked_single().shape, ())

    a = Pair([(1,2),(4,5)])
    self.assertEqual(a.masked_single(), Pair.MASKED)
    self.assertEqual(type(a.masked_single()), Pair)
    self.assertEqual(a.masked_single().shape, ())

    a = Vector3([(1,2,3),(4,5,6)])
    self.assertEqual(a.masked_single(), Vector3.MASKED)
    self.assertEqual(type(a.masked_single()), Vector3)
    self.assertEqual(a.masked_single().shape, ())

    a = Quaternion([(1,2,3,4),(4,5,6,7)])
    self.assertEqual(a.masked_single(), Quaternion.MASKED)
    self.assertEqual(type(a.masked_single()), Quaternion)
    self.assertEqual(a.masked_single().shape, ())

    a = Matrix([(1,2),(4,5)])
    self.assertEqual(a.masked_single(), Matrix.MASKED2)
    self.assertEqual(type(a.masked_single()), Matrix)
    self.assertEqual(a.masked_single().shape, ())

    a = Matrix([(1,2,3),(4,5,6),(7,8,9)])
    self.assertEqual(a.masked_single(), Matrix3.MASKED3)
    self.assertEqual(type(a.masked_single()), Matrix)
    self.assertEqual(a.masked_single().shape, ())

    a = Matrix3([(1,2,3),(4,5,6),(7,8,9)])
    self.assertEqual(a.masked_single(), Matrix3.MASKED)
    self.assertEqual(type(a.masked_single()), Matrix3)
    self.assertEqual(a.masked_single().shape, ())
  #=============================================================================



#*******************************************************************************



############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
