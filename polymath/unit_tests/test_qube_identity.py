################################################################################
# Tests for Qube.identity() for all subclasses.
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_identity(unittest.TestCase):

  # runTest
  def runTest(self):

    a = Scalar((1,2,3))
    self.assertEqual(a.identity(), 1)
    self.assertEqual(type(a.identity()), Scalar)
    self.assertEqual(type(a.identity().values), int)
    self.assertEqual(a.identity().shape, ())

    a = Scalar((1.,2.,3.))
    self.assertEqual(a.identity(), 1)
    self.assertEqual(type(a.identity()), Scalar)
    self.assertEqual(type(a.identity().values), float)
    self.assertEqual(a.identity().shape, ())

    a = Boolean([True,False])
    self.assertEqual(a.identity(), True)

    a = Vector([(1,2,3),(4,5,6)])
    self.assertRaises(TypeError, a.identity)

    a = Pair([(1,2),(4,5)])
    self.assertRaises(TypeError, a.identity)

    a = Vector3([(1,2,3),(4,5,6)])
    self.assertRaises(TypeError, a.identity)

    a = Quaternion([(1,2,3,4),(4,5,6,7)])
    self.assertEqual(a.identity(), (1,0,0,0))
    self.assertEqual(type(a.identity()), Quaternion)
    self.assertEqual(a.identity().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.identity().shape, ())

    a = Quaternion([(1.,2.,3.,4.),(4.,5.,6.,7.)])
    self.assertEqual(a.identity(), (1,0,0,0))
    self.assertEqual(type(a.identity()), Quaternion)
    self.assertEqual(a.identity().values.dtype, np.dtype('float'))
    self.assertEqual(a.identity().shape, ())

    a = Matrix([(1,2),(4,5)])
    self.assertEqual(a.identity(), [(1,0),(0,1)])
    self.assertEqual(type(a.identity()), Matrix)
    self.assertEqual(a.identity().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.identity().shape, ())

    a = Matrix([(1,2,3),(4,5,6),(7,8,9)])
    self.assertEqual(a.identity(), [(1,0,0),(0,1,0),(0,0,1)])
    self.assertEqual(type(a.identity()), Matrix)
    self.assertEqual(a.identity().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.identity().shape, ())

    a = Matrix3([(1,2,3),(4,5,6),(7,8,9)])
    self.assertEqual(a.identity(), [(1,0,0),(0,1,0),(0,0,1)])
    self.assertEqual(type(a.identity()), Matrix3)
    self.assertEqual(a.identity().values.dtype, np.dtype('float'))  # coerced
    self.assertEqual(a.identity().shape, ())

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
