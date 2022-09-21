################################################################################
# Boolean tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Boolean, Scalar, Vector, Units

class Test_Boolean(unittest.TestCase):

  def runTest(self):

    np.random.seed(7768)

    ############################################################################
    # Constructor
    ############################################################################

    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    self.assertEqual(a,b)

    b = Boolean(a)
    self.assertEqual(a,b)

    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    self.assertEqual(a[~mask], values[~mask])

    self.assertTrue(np.all(a.as_mask_where_nonzero() == a.values & ~mask))
    self.assertTrue(np.all(a.as_mask_where_zero() == ~a.values & ~mask))
    self.assertTrue(np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask))
    self.assertTrue(np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask))

    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    self.assertEqual(a, values)

    self.assertTrue(a == a.as_mask_where_nonzero())
    self.assertTrue(~a == a.as_mask_where_zero())
    self.assertTrue(a == a.as_mask_where_nonzero_or_masked())
    self.assertTrue(~a == a.as_mask_where_zero_or_masked())

    self.assertEqual(Boolean(True, True), Boolean.MASKED)
    self.assertEqual(Boolean(True, False), True)
    self.assertEqual(Boolean(False, False), False)
    self.assertEqual(Boolean(False, True), Boolean.MASKED)

    a = Boolean(N//2 * [True] + N//2 * [False])
    self.assertEqual(a[:N//2], True)
    self.assertEqual(a[N//2:], False)

    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    self.assertEqual(a[~b], 0.)
    self.assertTrue((a[b] != 0.).all())

    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    self.assertEqual(b, (a.data != 0.))

    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    self.assertEqual(b[a.mask], Boolean.MASKED)
    self.assertTrue(np.all(b[a.data == 0.].as_mask_where_nonzero() == False))

    ############################################################################
    # Disallowed base class operations
    ############################################################################

    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    self.assertRaises(TypeError, a.set_units, Units.KM)

    da_dt = Boolean(np.random.randn(N))
    self.assertRaises(TypeError, a.insert_deriv, 't', da_dt)

    self.assertRaises(TypeError, Boolean, a.values, units=Units.KM)

    ############################################################################
    # as_boolean
    ############################################################################

    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean.as_boolean(a)
    self.assertTrue(a is b)

    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean.as_boolean(Scalar(a))
    self.assertFalse(a is b)
    self.assertEqual(b[a.mask], Boolean.MASKED)
    self.assertTrue(np.all(b[a.data == 0.].as_mask_where_nonzero() == False))

    a = Boolean.as_boolean(True)
    self.assertEqual(a, True)
    self.assertEqual(type(a), Boolean)

    a = Boolean.as_boolean(False)
    self.assertEqual(a, False)
    self.assertEqual(type(a), Boolean)

    a = Boolean.as_boolean(2)
    self.assertEqual(a, True)
    self.assertEqual(type(a), Boolean)

    a = Boolean.as_boolean(0)
    self.assertEqual(a, False)
    self.assertEqual(type(a), Boolean)

    a = Boolean.as_boolean(-2.)
    self.assertEqual(a, True)
    self.assertEqual(type(a), Boolean)

    a = Boolean.as_boolean(0.)
    self.assertEqual(a, False)
    self.assertEqual(type(a), Boolean)

    ############################################################################
    # as_int()
    ############################################################################

    N = 100
    a = Boolean(np.random.randn(N) < 0.)

    c = a.as_int()
    self.assertEqual(c, a)
    self.assertEqual(c[a], 1)
    self.assertEqual(c[~a], 0)
    self.assertEqual(type(c), Scalar)
    self.assertEqual(c.values.dtype, np.dtype('int'))

    self.assertFalse(a.readonly)
    self.assertFalse(c.readonly)

    self.assertTrue(a.as_readonly().readonly)
    self.assertFalse((~a.as_readonly()).readonly)

    a = Boolean(True)
    c = a.as_int()
    self.assertEqual(c, a)
    self.assertEqual(c, 1)
    self.assertEqual(type(c.values), int)

    a = Boolean(False)
    c = a.as_int()
    self.assertEqual(c, a)
    self.assertEqual(c, 0)
    self.assertEqual(type(c.values), int)

    ############################################################################
    # as_float()
    ############################################################################

    N = 100
    a = Boolean(np.random.randn(N) < 0.)

    c = a.as_float()
    self.assertEqual(c, a)
    self.assertEqual(c[a], 1.)
    self.assertEqual(c[~a], 0.)
    self.assertEqual(type(c), Scalar)
    self.assertEqual(c.values.dtype, np.dtype('float'))

    self.assertFalse(a.readonly)
    self.assertFalse(c.readonly)

    self.assertTrue(a.as_readonly().readonly)
    self.assertFalse((~a.as_readonly()).readonly)

    a = Boolean(True)
    c = a.as_float()
    self.assertEqual(c, a)
    self.assertEqual(c, 1.)
    self.assertEqual(type(c.values), float)

    a = Boolean(False)
    c = a.as_float()
    self.assertEqual(c, a)
    self.assertEqual(c, 0.)
    self.assertEqual(type(c.values), float)

    ############################################################################
    # ~ operator (not)
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = ~a

    self.assertEqual(b[0], True)
    self.assertEqual(b[1], Boolean.MASKED)
    self.assertEqual(b[2], Boolean.MASKED)
    self.assertEqual(b[3], False)

    N = 100
    a = Boolean(np.random.randn(N) < 0.)

    c = ~a
    self.assertEqual(c, np.logical_not(a.values))

    self.assertFalse(a.readonly)
    self.assertFalse(c.readonly)

    self.assertTrue(a.as_readonly().readonly)
    self.assertFalse((~a.as_readonly()).readonly)

    ############################################################################
    # & operator (and)
    #
    # Truth table for three-valued logic
    #           False       Masked      True
    # False     False       False       False
    # Masked    False       Masked      Masked
    # True      False       Masked      True
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = a.tvl_and(b)

    self.assertEqual(ab[0],   False)
    self.assertEqual(ab[:,0], False)
    self.assertEqual(ab[3,3], True)
    self.assertEqual(ab[1:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,1:], Boolean.MASKED)

    ab = a & b

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[0,3], False)
    self.assertEqual(ab[3,0], False)
    self.assertEqual(ab[3,3], True)
    self.assertEqual(ab[:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:], Boolean.MASKED)

    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(np.random.randn(4,N) < 0.5)

    c = a & b
    self.assertEqual(c, a.values & b.values)
    self.assertTrue((c == a.values & b.values).all())

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(c.readonly)

    self.assertTrue(a.as_readonly().readonly)
    self.assertTrue(b.as_readonly().readonly)
    self.assertFalse((a.as_readonly() & b.as_readonly()).readonly)

    self.assertFalse((a.as_readonly() & b).readonly)
    self.assertFalse((a & b.as_readonly()).readonly)

    c = a & False
    self.assertEqual(c, False)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    c = a & True
    self.assertEqual(c, a)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    c = a & (N * [True])
    self.assertEqual(c, a)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    ############################################################################
    # | operator (or)
    #
    # Truth table for three-valued logic
    #               False       Masked(F)   Masked(T)   True
    # False         False       Masked      Masked      True
    # Masked(F)     Masked      Masked      Masked      True
    # Masked(T)     Masked      Masked      Masked      True
    # True          True        True        True        True
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = a.tvl_or(b)

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[:,3], True)
    self.assertEqual(ab[3,:], True)
    self.assertEqual(ab[:3,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:3], Boolean.MASKED)

    ab = a | b

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[0,3], True)
    self.assertEqual(ab[3,0], True)
    self.assertEqual(ab[3,3], True)
    self.assertEqual(ab[:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:], Boolean.MASKED)

    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(np.random.randn(4,N) < 0.5)

    c = a | b
    self.assertEqual(c, a.values | b.values)

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(c.readonly)

    self.assertTrue(a.as_readonly().readonly)
    self.assertTrue(b.as_readonly().readonly)
    self.assertFalse((a.as_readonly() | b.as_readonly()).readonly)

    self.assertFalse((a.as_readonly() | b).readonly)
    self.assertFalse((a | b.as_readonly()).readonly)

    c = a | False
    self.assertEqual(c, a)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    c = a | True
    self.assertEqual(c, True)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    ############################################################################
    # ^ operator (xor)
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = a ^ b

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[3,3], False)
    self.assertEqual(ab[0,3], True)
    self.assertEqual(ab[3,0], True)
    self.assertEqual(ab[:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:], Boolean.MASKED)

    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(np.random.randn(4,N) < 0.5)

    c = a ^ b
    self.assertEqual(c, a.values ^ b.values)
    self.assertTrue(c == (a.values ^ b.values))

    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(c.readonly)

    self.assertTrue(a.as_readonly().readonly)
    self.assertTrue(b.as_readonly().readonly)
    self.assertFalse((a.as_readonly() ^ b.as_readonly()).readonly)

    self.assertFalse((a.as_readonly() ^ b).readonly)
    self.assertFalse((a ^ b.as_readonly()).readonly)

    c = a ^ False
    self.assertEqual(c, a)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    c = a ^ True
    self.assertEqual(c, ~a)
    self.assertEqual(type(c), Boolean)
    self.assertEqual(c.shape, (N,))

    ############################################################################
    # &= operator
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]

    ab = Boolean(4*[[False, False, True, True]], 4*[[False, True, True, False]])
    ab &= b

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[0,3], False)
    self.assertEqual(ab[3,0], False)
    self.assertEqual(ab[3,3], True)
    self.assertEqual(ab[:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:], Boolean.MASKED)

    N = 100
    a = Boolean(np.random.randn(4,N) < 0.)
    b = Boolean(np.random.randn(N) < 0.5)

    c = a & b
    a &= b
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    b = (np.random.randn(N) < 0.5)
    c = a & b
    a &= b
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    c = a & True
    a &= True
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    c = a & False
    a &= False
    self.assertEqual(a, c)

    ############################################################################
    # |= operator
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]

    ab = Boolean(4*[[False, False, True, True]], 4*[[False, True, True, False]])
    ab |= b

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[0,3], True)
    self.assertEqual(ab[3,0], True)
    self.assertEqual(ab[3,3], True)
    self.assertEqual(ab[:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:], Boolean.MASKED)

    N = 100
    a = Boolean(np.random.randn(4,N) < 0.)
    b = Boolean(np.random.randn(N) < 0.5)

    c = a | b
    a |= b
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    b = (np.random.randn(N) < 0.5)
    c = a | b
    a |= b
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    c = a | 22.
    a |= 22.
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    c = a | False
    a |= False
    self.assertEqual(a, c)

    ############################################################################
    # ^= operator
    ############################################################################

    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]

    ab = Boolean(4*[[False, False, True, True]], 4*[[False, True, True, False]])
    ab ^= b

    self.assertEqual(ab[0,0], False)
    self.assertEqual(ab[3,3], False)
    self.assertEqual(ab[0,3], True)
    self.assertEqual(ab[3,0], True)
    self.assertEqual(ab[:,1:3], Boolean.MASKED)
    self.assertEqual(ab[1:3,:], Boolean.MASKED)

    N = 100
    a = Boolean(np.random.randn(4,N) < 0.)
    b = Boolean(np.random.randn(N) < 0.5)

    c = a ^ b
    a ^= b
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    b = (np.random.randn(N) < 0.5)
    c = a ^ b
    a ^= b
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    c = a ^ 22.
    a ^= True
    self.assertEqual(a, c)

    a = Boolean(np.random.randn(4,N) < 0.)
    c = a | 0
    a |= 0.
    self.assertEqual(a, c)

    ############################################################################
    # Other arithmetic
    ############################################################################

    # Confirm True == 1 in arithmetic
    a = Boolean(True) + 1
    self.assertEqual(a, 2)
    self.assertTrue(type(a), Scalar)

    a = 1 + Boolean(True)
    self.assertEqual(a, 2)
    self.assertTrue(type(a), Scalar)

    a = Boolean(True) - 2
    self.assertEqual(a, -1)
    self.assertTrue(type(a), Scalar)

    a = 3 - Boolean(True)
    self.assertEqual(a, 2)
    self.assertTrue(type(a), Scalar)

    a = Boolean(True) / 2
    self.assertEqual(a, 0.5)
    self.assertTrue(type(a), Scalar)

    a = 2 / Boolean(True)
    self.assertEqual(a, 2)
    self.assertTrue(type(a), Scalar)

    a = Boolean(True) // 1
    self.assertEqual(a, 1)
    self.assertTrue(type(a), Scalar)

    a = 2 // Boolean(True)
    self.assertEqual(a, 2)
    self.assertTrue(type(a), Scalar)

    a = Boolean(True) % 2
    self.assertEqual(a, 1)
    self.assertTrue(type(a), Scalar)

    a = 2 % Boolean(True)
    self.assertEqual(a, 0)
    self.assertTrue(type(a), Scalar)

    # Confirm False == 0 in arithmetic
    a = Boolean(False) + 1
    self.assertEqual(a, 1)
    self.assertTrue(type(a), Scalar)

    a = 1 + Boolean(False)
    self.assertEqual(a, 1)
    self.assertTrue(type(a), Scalar)

    a = Boolean(False) - 1
    self.assertEqual(a, -1)
    self.assertTrue(type(a), Scalar)

    a = 3 - Boolean(False)
    self.assertEqual(a, 3)
    self.assertTrue(type(a), Scalar)

    a = Boolean(False) / 2
    self.assertEqual(a, 0)
    self.assertTrue(type(a), Scalar)

    a = 2 / Boolean(False)
    self.assertTrue(a.mask)
    self.assertTrue(type(a), Scalar)

    a = Boolean(False) // 1
    self.assertEqual(a, 0)
    self.assertTrue(type(a), Scalar)

    a = 2 // Boolean(False)
    self.assertTrue(a.mask)
    self.assertTrue(type(a), Scalar)

    a = Boolean(False) % 2
    self.assertEqual(a, 0)
    self.assertTrue(type(a), Scalar)

    a = 2 % Boolean(False)
    self.assertTrue(a.mask)
    self.assertTrue(type(a), Scalar)

    # Test tuples
    a = Boolean((True,False)) + 1
    self.assertEqual(Boolean((True,False)) + 1, (2,1))
    self.assertEqual(1 + Boolean((True,False)), (2,1))
    self.assertEqual(Boolean((True,False)) - 1, (0,-1))
    self.assertEqual(1 - Boolean((True,False)), (0,1))
    self.assertEqual(Boolean((True,False)) * 2, (2,0))
    self.assertEqual(2 * Boolean((True,False)), (2,0))
    self.assertEqual(Boolean((True,False)) / 1, (1,0))
    self.assertEqual((1 / Boolean((True,False))).mask[0], False)
    self.assertEqual((1 / Boolean((True,False))).mask[1], True)
    self.assertEqual(Boolean((True,False)) // 1, (1,0))
    self.assertEqual((1 // Boolean((True,False))).mask[0], False)
    self.assertEqual((1 // Boolean((True,False))).mask[1], True)
    self.assertEqual(Boolean((True,False)) % 1, (0,0))
    self.assertEqual((1 % Boolean((True,False))).mask[0], False)
    self.assertEqual((1 % Boolean((True,False))).mask[1], True)

    ############################################################################
    # More masking
    ############################################################################

    N = 200
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)

    mask = a.as_mask_where_nonzero()
    self.assertTrue(a[mask].all())
    self.assertTrue((a[mask] == True).all())

    mask = a.as_mask_where_zero()
    self.assertTrue(not a[mask].any())
    self.assertTrue((a[mask] == False).all())

    mask = a.as_mask_where_nonzero_or_masked()
    self.assertTrue(not (a[mask] == False).any())

    mask = a.as_mask_where_zero_or_masked()
    self.assertTrue(not (a[mask] == True).any())

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
