################################################################################
# Tests for Qube.as_this_type() for all subclasses
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_as_this_type(unittest.TestCase):

  # runTest
  def runTest(self):

    # Scalar, int
    a = Scalar((1,2,3))

    b = a.as_this_type(7)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_int())

    b = a.as_this_type(7., coerce=True)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_int())

    b = a.as_this_type(7., coerce=False)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_float())

    b = a.as_this_type(Qube(7.), coerce=True)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_int())

    b = a.as_this_type(Qube(7.), coerce=False)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_float())

    b = Scalar(7)
    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    b = Scalar(7.)
    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is not bb)

    b = Boolean(True)
    bb = a.as_this_type(b, coerce=False)
    self.assertEqual(bb, 1)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_int())

    b = Scalar((7,8,9))
    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    b = Scalar((7.,8.,9.))
    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is not bb)

    b = Boolean([True,False,False,True])
    bb = a.as_this_type(b, coerce=False)
    self.assertEqual(bb, [1,0,0,1])
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_int())

    # Scalar, float
    a = Scalar(1.)

    b = a.as_this_type(7., coerce=True)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_float())

    b = a.as_this_type(7, coerce=True)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_float())

    b = a.as_this_type(7, coerce=False)
    self.assertEqual(b, 7)
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_int())

    b = Scalar(7.)
    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    b = Scalar(7)
    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is not bb)

    b = Boolean(True)
    bb = a.as_this_type(b, coerce=False)
    self.assertEqual(bb, 1)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_int())

    bb = a.as_this_type(b, coerce=True)
    self.assertEqual(bb, 1)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_float())

    b = Scalar((7.,8.,9.))
    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    b = Scalar((7,8,9))
    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)

    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is not bb)

    b = Boolean([True,False,False,True])
    bb = a.as_this_type(b, coerce=False)
    self.assertEqual(bb, [1,0,0,1])
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_int())

    bb = a.as_this_type(b, coerce=True)
    self.assertEqual(bb, [1,0,0,1])
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_float())

    # Scalar, derivs
    a = Scalar(1.)

    b = Scalar(7)
    db_dt = Scalar(np.arange(4.).reshape(2,2), drank=2)
    b.insert_deriv('t', db_dt)

    bb = a.as_this_type(b, recursive=False, coerce=True)
    self.assertEqual(bb, 7)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_float())
    self.assertEqual(bb.derivs, {})

    bb = a.as_this_type(b, recursive=False, coerce=False)
    self.assertEqual(bb, 7)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(bb.is_int())
    self.assertEqual(bb.derivs, {})

    bb = a.as_this_type(b, recursive=True, coerce=True)
    self.assertEqual(bb, 7)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(type(bb.d_dt), Scalar)
    self.assertTrue(bb.is_float())
    self.assertTrue(bb.d_dt.is_float())

    bb = a.as_this_type(b, recursive=True, coerce=False)
    self.assertEqual(bb, 7)
    self.assertTrue(type(bb), Scalar)
    self.assertTrue(type(bb.d_dt), Scalar)
    self.assertTrue(bb.is_int())
    self.assertTrue(bb.d_dt.is_float())

    # Boolean
    a = Boolean((True,False))

    b = a.as_this_type(7)
    self.assertEqual(b, True)
    self.assertTrue(type(b), Boolean)
    self.assertTrue(b.is_bool())

    b = a.as_this_type(7., coerce=True)
    self.assertEqual(b, True)
    self.assertTrue(type(b), Boolean)
    self.assertTrue(b.is_bool())

    b = a.as_this_type(7., coerce=False)
    self.assertEqual(b, True)
    self.assertTrue(type(b), Boolean)
    self.assertTrue(b.is_bool())

    b = a.as_this_type(Scalar([7.,0.]), coerce=True)
    self.assertEqual(b, [True,False])
    self.assertTrue(type(b), Boolean)

    b = a.as_this_type(Scalar([7.,0.]), coerce=False)
    self.assertEqual(b, [True,False])
    self.assertTrue(type(b), Boolean)

    # Vector
    a = Vector((1.,2.,3.))

    self.assertRaises(ValueError, a.as_this_type, 7)

    b = Scalar((1.,2.,3.))
    self.assertRaises(ValueError, a.as_this_type, b)

    b = Boolean((False,True,False))
    self.assertRaises(ValueError, a.as_this_type, b)

    b = Vector((1.,2.,3.))
    bb = a.as_this_type(b)
    self.assertEqual(type(bb), Vector)

    b = Vector3((1.,2.,3.))
    bb = a.as_this_type(b)
    self.assertEqual(type(bb), Vector)

    b = Pair((1.,2.))
    bb = a.as_this_type(b)
    self.assertEqual(type(bb), Vector)

    # Vector3
    a = Vector3((1.,2.,3.))

    self.assertRaises(ValueError, a.as_this_type, 7)

    b = Scalar((1.,2.,3.))
    self.assertRaises(ValueError, a.as_this_type, b)

    b = Boolean((False,True,False))
    self.assertRaises(ValueError, a.as_this_type, b)

    b = Vector((1.,2.,3.))
    bb = a.as_this_type(b)
    self.assertEqual(type(bb), Vector3)

    b = Vector3((1.,2.,3.))
    bb = a.as_this_type(b)
    self.assertTrue(b is bb)

    b = Pair((1.,2.))
    self.assertRaises(ValueError, a.as_this_type, b)

    b = Vector3((1.,2.,3.))
    db_dt = Vector3(np.arange(6.).reshape(3,2), drank=1)
    b.insert_deriv('t', db_dt)
    bb = a.as_this_type(b, recursive=True)
    self.assertTrue(b is bb)

    b = Vector((1.,2.,3.))
    db_dt = Vector(np.arange(6.).reshape(3,2), drank=1)
    b.insert_deriv('t', db_dt)
    bb = a.as_this_type(b, recursive=True)
    self.assertTrue(b is not bb)
    self.assertTrue(np.all(bb.values == b.values))
    self.assertTrue(np.all(bb.d_dt.values == b.d_dt.values))
    self.assertTrue(type(b), Vector3)
    self.assertTrue(type(b.d_dt), Vector3)

    # read-only status
    a = Scalar(1.)

    b = Scalar((1,2,3))
    b.as_readonly()

    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)
    self.assertTrue(bb.readonly)

    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is not bb)
    self.assertTrue(not bb.readonly)

    a = Pair((1.,2.))

    b = Pair((2,3))
    db_dt = Pair(np.arange(4.).reshape(2,2), drank=1)
    b.as_readonly()
    b.insert_deriv('t', db_dt)
    self.assertTrue(b.d_dt.readonly)

    bb = a.as_this_type(b, coerce=False)
    self.assertTrue(b is bb)
    self.assertTrue(bb.readonly)

    bb = a.as_this_type(b, coerce=True)
    self.assertTrue(b is not bb)
    self.assertTrue(not bb.readonly)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
