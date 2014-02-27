################################################################################
# Vector tests for arithmetic operations
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Vector, Scalar, Boolean, Units

class Test_Vector_ops(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # Unary plus
    ############################################################################

    a = Vector((1,2,3))
    b = +a
    self.assertEqual(b, (1,2,3))
    self.assertEqual(type(b), Vector)
    self.assertTrue(b.is_int())
    self.assertFalse(b.is_float())

    a = Vector((1.,2.,3.))
    b = +a
    self.assertEqual(b, (1,2,3))
    self.assertEqual(type(b), Vector)
    self.assertFalse(b.is_int())
    self.assertTrue(b.is_float())

    a = Vector((1,2))
    b = +a
    self.assertEqual(b, (1,2))
    self.assertEqual(type(b), Vector)
    self.assertTrue(b.is_int())
    self.assertFalse(b.is_float())

    a = Vector((1.,2.))
    b = +a
    self.assertEqual(b, (1,2))
    self.assertEqual(type(b), Vector)
    self.assertFalse(b.is_int())
    self.assertTrue(b.is_float())

    # Derivatives, readonly
    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))})
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (1,1,2))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))}).as_readonly()
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (1,1,2))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__iadd__, (1,1,1))
    self.assertRaises(ValueError, b.__iadd__, (1,1,1))

    a = Vector((1,2), derivs={'t':Vector((3,4))})
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,4))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2), derivs={'t':Vector((3,4))}).as_readonly()
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,4))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__iadd__, 1)
    self.assertRaises(ValueError, b.__iadd__, 1)

    ############################################################################
    # Unary minus
    ############################################################################

    a = Vector((1,2,3))
    b = -a
    self.assertEqual(b, (-1,-2,-3))
    self.assertEqual(type(b), Vector)
    self.assertTrue(b.is_int())

    a = Vector((1.,2.,3.))
    b = -a
    self.assertEqual(b, (-1.,-2.,-3.))
    self.assertEqual(type(b), Vector)
    self.assertTrue(b.is_float())

    a = Vector((1,2))
    b = -a
    self.assertEqual(b, (-1,-2))
    self.assertEqual(type(b), Vector)
    self.assertTrue(b.is_int())

    a = Vector((1.,2.))
    b = -a
    self.assertEqual(b, (-1,-2))
    self.assertEqual(type(b), Vector)
    self.assertTrue(b.is_float())

    # Derivatives, readonly
    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))})
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-1,-1,-2))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))}).as_readonly()
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-1,-1,-2))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__isub__, (1,1,1))
    self.assertRaises(ValueError, b.__isub__, (1,1,1))

    a = Vector((1,2), derivs={'t':Vector((3,4))})
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-3,-4))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2), derivs={'t':Vector((3,4))}).as_readonly()
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-3,-4))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__isub__, 1)
    self.assertRaises(ValueError, b.__isub__, 1)

    ############################################################################
    # abs()
    ############################################################################

    a = Vector((3,4))
    self.assertRaises(TypeError, a.__abs__)

    ############################################################################
    # Addition
    ############################################################################

    a = Vector((1,2,3))
    self.assertRaises(TypeError, a.__add__, 1)  # rank mismatch

    expr = Vector((1,2,3)) + (1,2,3)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((1.,2.,3.)) + (1,2,3)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) + (1.,2.,3.)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) + Vector((1,2,3))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = (1.,2.,3.) + Vector((1.,2.,3.))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) + Vector((1.,2.,3.))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

# DOES NOT WORK! NumPy tries to iterate through array elements and perform
# 1 + Vector((1,2,3)) first. This fails.
#     expr = np.array((1,2,3)) + Vector((1,2,3))
#     self.assertEqual(expr, (2,4,6))
#     self.assertEqual(type(expr), Vector)
#     self.assertTrue(expr.is_int())

    expr = Vector([(1,2,3),(2,3,4)]) + (1,2,3)
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector([(1.,2.,3.),(2.,3.,4.)]) + (1,2,3)
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector([(1,2,3),(2,3,4)]) + (1.,2.,3.)
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) + Vector([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = (1,2,3) + Vector([(1.,2.,3.),(2.,3.,4.)])
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1.,2.,3.) + Vector([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) + ([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((1,2,3)) + ([(1.,2.,3.),(2.,3.,4.)])
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1.,2.,3.)) + ([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = ((1,2,3),(2,3,4)) + Vector((1,2,3))
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = ((1.,2.,3.),(2.,3.,4.)) + Vector((1,2,3))
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = ((1,2,3),(2,3,4)) + Vector((1.,2.,3.))
    self.assertEqual(expr, ((2,4,6),(3,5,7)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a + (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,2,1))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = (1,2,3) + a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,2,1))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a + (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,2,1))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a + [(1,2,3),(4,5,6)]
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, ((3,2,1),(3,2,1))) # d_dt must be broadcasted
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    # In-place
    a = Vector((1,2))
    a += (1,1)
    self.assertEqual(a, (2,3))

    a += (2,3)
    self.assertEqual(a, (4,6))
    self.assertTrue(a.is_int())

    a += (0.5,1.5)
    self.assertEqual(a, (4,7))  # no automatic conversion to float

    a = Vector([(1,2),(3,4)])
    b = Vector([(1,2),(3,4)], mask=(False,True))
    a += b
    self.assertEqual(a[0], (2,4))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector([(1,2),(3,4)])
    b = Vector((1,2), derivs={'t':Vector([(1,1),(2,2)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a += b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, [(2,4),(4,6)])
    self.assertEqual(a.d_dt, ((1,1),(2,2)))

    b = Vector((1,2), derivs={'t':Vector((1,2), drank=0)})
    a_copy = a.copy()
    self.assertRaises(ValueError, a.__iadd__, b)    # shape mismatch in deriv
    self.assertEqual(a, a_copy)                     # but object unchanged

    a = Vector((1,2), derivs={'t':Vector(((1,2),(3,4)), drank=1)})
    b = Vector((3,4), derivs={'t':Vector(((4,3),(2,1)), drank=1)})
    a += b
    self.assertEqual(a, (4,6))
    self.assertEqual(a.d_dt, ((5,5),(5,5)))

    ############################################################################
    # Subtraction
    ############################################################################

    a = Vector((1,2,3))
    self.assertRaises(TypeError, a.__add__, 1)  # rank mismatch

    expr = Vector((1,2,3)) - (1,2,3)
    self.assertEqual(expr, (0,0,0))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((1.,2.,3.)) - (1,2,3)
    self.assertEqual(expr, (0,0,0))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) - (1.,2.,3.)
    self.assertEqual(expr, (0,0,0))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) - Vector((1,2,3))
    self.assertEqual(expr, (0,0,0))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = (1.,2.,3.) - Vector((1.,2.,3.))
    self.assertEqual(expr, (0,0,0))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) - Vector((1.,2.,3.))
    self.assertEqual(expr, (0,0,0))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

# DOES NOT WORK! NumPy tries to iterate through array elements and perform
# 1 + Vector((1,2,3)) first. This fails.
#     expr = np.array((1,2,3)) - Vector((1,2,3))
#     self.assertEqual(expr, (0,0,0))
#     self.assertEqual(type(expr), Vector)
#     self.assertTrue(expr.is_int())

    expr = Vector([(1,2,3),(2,3,4)]) - (1,2,3)
    self.assertEqual(expr, ((0,0,0),(1,1,1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector([(1.,2.,3.),(2.,3.,4.)]) - (1,2,3)
    self.assertEqual(expr, ((0,0,0),(1,1,1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector([(1,2,3),(2,3,4)]) - (1.,2.,3.)
    self.assertEqual(expr, ((0,0,0),(1,1,1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) - Vector([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((0,0,0),(-1,-1,-1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = (1,2,3) - Vector([(1.,2.,3.),(2.,3.,4.)])
    self.assertEqual(expr, ((0,0,0),(-1,-1,-1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1.,2.,3.) - Vector([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((0,0,0),(-1,-1,-1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) - ([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((0,0,0),(-1,-1,-1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((1,2,3)) - ([(1.,2.,3.),(2.,3.,4.)])
    self.assertEqual(expr, ((0,0,0),(-1,-1,-1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1.,2.,3.)) - ([(1,2,3),(2,3,4)])
    self.assertEqual(expr, ((0,0,0),(-1,-1,-1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = ((1,2,3),(2,3,4)) - Vector((1,2,3))
    self.assertEqual(expr, ((0,0,0),(1,1,1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = ((1.,2.,3.),(2.,3.,4.)) - Vector((1,2,3))
    self.assertEqual(expr, ((0,0,0),(1,1,1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = ((1,2,3),(2,3,4)) - Vector((1.,2.,3.))
    self.assertEqual(expr, ((0,0,0),(1,1,1)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a - (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,2,1))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = (1,2,3) - a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-3,-2,-1))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a - (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,2,1))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a - [(1,2,3),(4,5,6)]
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, ((3,2,1),(3,2,1))) # d_dt must be broadcasted
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    # In-place
    a = Vector((1,2))
    a -= (1,1)
    self.assertEqual(a, (0,1))

    a -= (-2,-3)
    self.assertEqual(a, (2,4))
    self.assertTrue(a.is_int())

    a -= (0.5,1.5)
    self.assertEqual(a, (1,2))  # no automatic conversion to float

    a = Vector([(1,2),(3,4)])
    b = Vector([(1,2),(3,4)], mask=(False,True))
    a -= b
    self.assertEqual(a[0], (0,0))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector([(1,2),(3,4)])
    b = Vector((1,2), derivs={'t':Vector([(1,1),(2,2)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a -= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, [(0,0),(2,2)])
    self.assertEqual(a.d_dt, ((-1,-1),(-2,-2)))

    b = Vector((1,2), derivs={'t':Vector((1,2), drank=0)})
    a_copy = a.copy()
    self.assertRaises(ValueError, a.__iadd__, b)    # shape mismatch in deriv
    self.assertEqual(a, a_copy)                     # but object unchanged

    a = Vector((1,2), derivs={'t':Vector(((1,2),(3,4)), drank=1)})
    b = Vector((3,4), derivs={'t':Vector(((4,3),(2,1)), drank=1)})
    a -= b
    self.assertEqual(a, (-2,-2))
    self.assertEqual(a.d_dt, ((-3,-1),(1,3)))

    ############################################################################
    # Multiplication
    ############################################################################

    expr = Vector((1,2,3)) * 2
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((1.,2.,3.)) * 2
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) * 2.
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = 2 * Vector((1,2,3))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = 2 * Vector((1.,2.,3.))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = 2. * Vector((1,2,3))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) * (1,2)
    self.assertEqual(expr, [(1,2,3),(2,4,6)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((1.,2.,3.)) * (1,2)
    self.assertEqual(expr, [(1,2,3),(2,4,6)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((1,2,3)) * (1.,2.)
    self.assertEqual(expr, [(1,2,3),(2,4,6)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1,2) * Vector((1,2,3))
    self.assertEqual(expr, [(1,2,3),(2,4,6)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = (1,2) * Vector((1.,2.,3.))
    self.assertEqual(expr, [(1,2,3),(2,4,6)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = (1.,2.) * Vector((1,2,3))
    self.assertEqual(expr, [(1,2,3),(2,4,6)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector([(1,2,3),(2,3,4)]) * (1,2)
    self.assertEqual(expr, ((1,2,3),(4,6,8)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector([(1.,2.,3.),(2.,3.,4.)]) * (1,2)
    self.assertEqual(expr, ((1,2,3),(4,6,8)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector([(1,2,3),(2,3,4)]) * (1.,2.)
    self.assertEqual(expr, ((1,2,3),(4,6,8)))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a * 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (6,4,2))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = 2 * a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (6,4,2))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a * (1,2)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(3,2,1),(6,4,2)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = (1,2) * a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(3,2,1),(6,4,2)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a * 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (6,4,2))
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    a = Scalar((1,3), derivs={'t':Scalar((1,2))})
    b = Vector((1,2), derivs={'t':Vector((3,2))})
    c = b * a
    # (1*(1,2), 3*(1,2))
    self.assertEqual(c, [(1,2),(3,6)])
    # [(1*3+1*1,1*2+2*1),(3*3+1*2,3*2+2*2)]
    self.assertEqual(c.d_dt, [(4,4),(11,10)])

    c = a * b
    self.assertEqual(c, [(1,2),(3,6)])
    self.assertEqual(c.d_dt, [(4,4),(11,10)])

    # In-place
    a = Vector((1,2))
    a *= 2
    self.assertEqual(a, (2,4))

    a *= 0.25
    self.assertEqual(a, (0,1))  # no automatic conversion to float

    a = Vector([(1,2),(3,4)])
    b = (2,3)
    a *= b
    self.assertEqual(a, [(2,4),(9,12)])

    a = Vector([(1,2),(3,4)])
    b = Scalar((2,3), mask=(False,True))
    a *= b
    self.assertEqual(a[0], (2,4))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector([(1,2),(3,4)])
    b = Scalar(2, derivs={'t':Scalar(1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a *= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, [(2,4),(6,8)])
    self.assertEqual(a.d_dt, ((1,2),(3,4)))

    a = Vector((3,4), derivs={'t':Vector((2,1), drank=0)})
    b = Scalar(2, derivs={'t':Scalar(1)})
    a *= b
    self.assertEqual(a, (6,8))
    self.assertEqual(a.d_dt, (7,6))

    ############################################################################
    # Division
    ############################################################################

    expr = Vector((2,4,6)) / 2
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,6)) / (1,2)
    self.assertEqual(expr, [(2,4,6),(1,2,3)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Vector((2,4,6), derivs={'t':Vector((6,4,2))})
    b = a / 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,2,1))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Vector((2,4,6))
    b = Scalar(2, derivs={'t':Scalar(-2)})
    c = a / b
    self.assertEqual(c, (1,2,3))
    self.assertEqual(c.d_dt, (1,2,3))
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))})
    b = Scalar(2, derivs={'t':Scalar(-2)})
    c = a / b
    self.assertEqual(c, (1,2,3))
    self.assertEqual(c.d_dt, -a/b/b*b.d_dt + a.d_dt/b)
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(-2)})
    c = a / b
    self.assertEqual(c, (1,2,3))
    self.assertEqual(c.d_dt, -a/b/b*b.d_dt + a.d_dt/b)
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))})
    b = Scalar(2, derivs={'t':Scalar(-2)}).as_readonly()
    c = a / b
    self.assertEqual(c, (1,2,3))
    self.assertEqual(c.d_dt, -a/b/b*b.d_dt + a.d_dt/b)
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(-2)}).as_readonly()
    c = a / b
    self.assertEqual(c, (1,2,3))
    self.assertEqual(c.d_dt, -a/b/b*b.d_dt + a.d_dt/b)
    self.assertTrue(c.readonly)
    self.assertTrue(c.d_dt.readonly)

    # In-place
    a = Vector((4,6))
    a /= 2
    self.assertEqual(a, (2,3))

    a = Vector((1,2))
    a /= 0.5
    self.assertEqual(a, (2,4))  # no automatic conversion to float
    self.assertTrue(a.is_int())

    a = Vector([(3,4),(4,6)])
    b = Scalar((1,2), mask=(False,True))
    a /= b
    self.assertEqual(a[0], (3,4))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector([(3,4),(4,6)])
    b = Scalar((1,2), mask=(False,False))
    a /= b
    self.assertEqual(a[0], (3,4))
    self.assertEqual(a[1], (2,3))

    a = Vector([(3,4),(4,6)])
    b = Scalar((1,2), mask=(False,True))
    a /= b
    self.assertEqual(a[0], (3,4))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector((9,-18))
    b = Scalar(3, derivs={'t':Scalar((1,2), drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    da_dt = -(a/b/b).without_derivs() * b.d_dt

    a /= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, (3,-6))

    DEL = 1.e-13
    self.assertAlmostEqual(a.d_dt.values[0,0], da_dt.values[0,0], delta=DEL)
    self.assertAlmostEqual(a.d_dt.values[0,1], da_dt.values[0,1], delta=DEL)
    self.assertAlmostEqual(a.d_dt.values[1,0], da_dt.values[1,0], delta=DEL)
    self.assertAlmostEqual(a.d_dt.values[1,1], da_dt.values[1,1], delta=DEL)

    a = Vector((9,-18))
    a /= 0
    self.assertTrue(a.mask)

    ############################################################################
    # Floor division
    ############################################################################

    expr = Vector((2,4,7)) // 2
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((2.,4.,7.)) // 2
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,7)) // 2.
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,7)) // (2,3)
    self.assertEqual(expr, [(1,2,3),(0,1,2)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((2.,4.,7.)) // (2,3)
    self.assertEqual(expr, [(1,2,3),(0,1,2)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,7)) // (2.,3.)
    self.assertEqual(expr, [(1,2,3),(0,1,2)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Vector((2,4,7), derivs={'t':Vector((6,4,2))})
    b = a // 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a // 2
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a // Scalar(2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a // Scalar(2).as_readonly()
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a // np.array(2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    # In-place
    a = Vector((4,7))
    a //= 2
    self.assertEqual(a, (2,3))

    a = Vector((5,8))
    a //= 3.5
    self.assertEqual(a, (1,2))  # no automatic conversion to float
    self.assertTrue(a.is_int())

    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,False))
    a //= b
    self.assertEqual(a, [(3,4),(2,3)])

    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,True))
    a //= b
    self.assertEqual(a[0], (3,4))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector([(3,4),(4,7)])
    b = Scalar((1,0))
    a //= b
    self.assertEqual(a[0], (3,4))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    ############################################################################
    # Modulus
    ############################################################################

    expr = Vector((2,4,7)) % 2
    self.assertEqual(expr, (0,0,1))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((2.,4.,7.)) % 2
    self.assertEqual(expr, (0,0,1))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,7)) % 2.
    self.assertEqual(expr, (0,0,1))
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,7)) % (2,3)
    self.assertEqual(expr, [(0,0,1),(2,1,1)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_int())

    expr = Vector((2.,4.,7.)) % (2,3)
    self.assertEqual(expr, [(0,0,1),(2,1,1)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    expr = Vector((2,4,7)) % (2.,3.)
    self.assertEqual(expr, [(0,0,1),(2,1,1)])
    self.assertEqual(type(expr), Vector)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Vector((2,4,7), derivs={'t':Vector((6,4,2))})
    b = a % 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a % 2
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a % Scalar(2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a % Scalar(2).as_readonly()
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)

    a = Vector((2,4,7)).as_readonly()
    b = a % np.array(2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    # In-place
    a = Vector((4,7))
    a %= 2
    self.assertEqual(a, (0,1))

    a = Vector((5,8))
    a %= 3.5
    self.assertEqual(a, (1,1))  # no automatic conversion to float
    self.assertTrue(a.is_int())

    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,False))
    a %= b
    self.assertEqual(a, [(0,0),(0,1)])

    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,True))
    a %= b
    self.assertEqual(a[0], (0,0))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Vector([(3,4),(4,7)])
    b = Scalar((1,0))
    a %= b
    self.assertEqual(a[0], (0,0))
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    ############################################################################
    # Reciprocal
    ############################################################################

    a = Vector((2,4,7))
    self.assertRaises(TypeError, a.reciprocal)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
