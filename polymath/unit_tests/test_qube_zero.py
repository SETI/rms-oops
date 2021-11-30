################################################################################
# Tests for Qube.zero() for all subclasses
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_zero(unittest.TestCase):

  def runTest(self):

    a = Scalar((1,2,3))
    self.assertEqual(a.zero(), 0)
    self.assertEqual(type(a.zero()), Scalar)
    self.assertEqual(type(a.zero().values), int)
    self.assertEqual(a.zero().shape, ())

    a = Scalar((1.,2.,3.))
    self.assertEqual(a.zero(), 0)
    self.assertEqual(type(a.zero()), Scalar)
    self.assertEqual(type(a.zero().values), float)
    self.assertEqual(a.zero().shape, ())

    a = Boolean([True,False])
    self.assertEqual(a.zero(), False)
    self.assertEqual(type(a.zero()), Boolean)
    self.assertEqual(type(a.zero().values), bool)
    self.assertEqual(a.zero().shape, ())

    a = Vector([(1,2,3),(4,5,6)])
    self.assertEqual(a.zero(), (0,0,0))
    self.assertEqual(type(a.zero()), Vector)
    self.assertEqual(a.zero().values.dtype, np.dtype('int'))
    self.assertEqual(a.zero().shape, ())

    a = Vector([(1.,2.,3.),(4.,5.,6.)])
    self.assertEqual(a.zero(), (0,0,0))
    self.assertEqual(type(a.zero()), Vector)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))
    self.assertEqual(a.zero().shape, ())

    a = Pair([(1,2),(4,5)])
    self.assertEqual(a.zero(), (0,0))
    self.assertEqual(type(a.zero()), Pair)
    self.assertEqual(a.zero().values.dtype, np.dtype('int'))
    self.assertEqual(a.zero().shape, ())

    a = Pair([(1.,2.),(4.,5.)])
    self.assertEqual(a.zero(), (0,0))
    self.assertEqual(type(a.zero()), Pair)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))
    self.assertEqual(a.zero().shape, ())

    a = Vector3([(1,2,3),(4,5,6)])
    self.assertEqual(a.zero(), (0,0,0))
    self.assertEqual(type(a.zero()), Vector3)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.zero().shape, ())

    a = Vector3([(1.,2.,3.),(4.,5.,6.)])
    self.assertEqual(a.zero(), (0,0,0))
    self.assertEqual(type(a.zero()), Vector3)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))
    self.assertEqual(a.zero().shape, ())

    a = Quaternion([(1,2,3,4),(4,5,6,7)])
    self.assertEqual(a.zero(), (0,0,0,0))
    self.assertEqual(type(a.zero()), Quaternion)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.zero().shape, ())

    a = Quaternion([(1.,2.,3.,4.),(4.,5.,6.,7.)])
    self.assertEqual(a.zero(), (0,0,0,0))
    self.assertEqual(type(a.zero()), Quaternion)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))
    self.assertEqual(a.zero().shape, ())

    a = Matrix([(1,2),(4,5)])
    self.assertEqual(a.zero(), [(0,0),(0,0)])
    self.assertEqual(type(a.zero()), Matrix)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.zero().shape, ())

    a = Matrix([(1.,2.),(4.,5.)])
    self.assertEqual(a.zero(), [(0,0),(0,0)])
    self.assertEqual(type(a.zero()), Matrix)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))
    self.assertEqual(a.zero().shape, ())

    a = Matrix3([(1,2,3),(4,5,6),(7,8,9)])
    self.assertEqual(a.zero(), [(0,0,0),(0,0,0),(0,0,0)])
    self.assertEqual(type(a.zero()), Matrix3)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.zero().shape, ())

    a = Matrix3([(1.,2.,3.),(4.,5.,6.),(7.,8.,9.)])
    self.assertEqual(a.zero(), [(0,0,0),(0,0,0),(0,0,0)])
    self.assertEqual(type(a.zero()), Matrix3)
    self.assertEqual(a.zero().values.dtype, np.dtype('float'))
    self.assertEqual(a.zero().shape, ())

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
