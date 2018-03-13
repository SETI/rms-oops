################################################################################
# Scalar tests for arithmetic and comparison operations
################################################################################

# from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Boolean, Units

class Test_Scalar_ops(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # Unary plus
    ############################################################################

    a = Scalar(1)
    b = +a
    self.assertEqual(b, 1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())
    self.assertFalse(b.is_float())

    a = Scalar(1.)
    b = +a
    self.assertEqual(b, 1)
    self.assertEqual(type(b), Scalar)
    self.assertFalse(b.is_int())
    self.assertTrue(b.is_float())

    a = Scalar((1,2))
    b = +a
    self.assertEqual(b, (1,2))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())
    self.assertFalse(b.is_float())

    a = Scalar((1.,2.))
    b = +a
    self.assertEqual(b, (1,2))
    self.assertEqual(type(b), Scalar)
    self.assertFalse(b.is_int())
    self.assertTrue(b.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__iadd__, 1)
    self.assertRaises(ValueError, b.__iadd__, 1)

    a = Scalar((1,2), derivs={'t':Scalar((3,4))})
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (3,4))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar((1,2), derivs={'t':Scalar((3,4))}).as_readonly()
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

    a = Scalar(1)
    b = -a
    self.assertEqual(b, -1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar(1.)
    b = -a
    self.assertEqual(b, -1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar((1,2))
    b = -a
    self.assertEqual(b, (-1,-2))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar((1.,2.))
    b = -a
    self.assertEqual(b, (-1,-2))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, -2)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, -2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__isub__, 1)
    #self.assertRaises(ValueError, b.__isub__, 1)
    b -= 1

    a = Scalar((1,2), derivs={'t':Scalar((3,4))})
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-3,-4))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar((1,2), derivs={'t':Scalar((3,4))}).as_readonly()
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (-3,-4))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__isub__, 1)
    #self.assertRaises(ValueError, b.__isub__, 1)
    b -= 1

    ############################################################################
    # abs()
    ############################################################################

    a = abs(Scalar(1))
    b = abs(a)
    self.assertEqual(b, 1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar(-1)
    b = abs(a)
    self.assertEqual(b, 1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar(1.)
    b = abs(a)
    self.assertEqual(b, 1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar(-1.)
    b = abs(a)
    self.assertEqual(b, 1)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar((1,-2))
    b = abs(a)
    self.assertEqual(b, (1,2))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar((-1.,2.))
    b = abs(a)
    self.assertEqual(b, (1,2))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = abs(a)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(b.d_dt.readonly)
    self.assertEqual(b.d_dt, 2)

    a = Scalar(-1, derivs={'t':Scalar(2)})
    b = abs(a)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(b.d_dt.readonly)
    self.assertEqual(b.d_dt, -2)

    a = Scalar((1,-1), derivs={'t':Scalar((2,2))})
    b = abs(a)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(b.d_dt.readonly)
    self.assertEqual(b.d_dt, (2,-2))

    a = Scalar(1).as_readonly()
    b = abs(a)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar((1,-1), derivs={'t':Scalar((2,2))}).as_readonly()
    b = abs(a)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(b.d_dt.readonly)

    ############################################################################
    # Addition
    ############################################################################

    expr = Scalar(1) + 1
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(1.) + 1
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(1) + 1.
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1 + Scalar(1)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 1. + Scalar(1)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1 + Scalar(1.)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((1,2,3)) + 1
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 1 + Scalar((1,2,3))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(1) + (1,2,3)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = (1,2,3) + Scalar(1)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(1) + np.array((1,2,3))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = np.array((1,2,3)) + Scalar(1)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar((1,2,3)) + 1.
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1. + Scalar((1,2,3))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((1.,2.,3.)) + 1
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1 + Scalar((1.,2.,3.))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(1) + (1.,2.,3.)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (1.,2.,3.) + Scalar(1)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(1.) + (1,2,3)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) + Scalar(1.)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a + (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)       # writeable because it is a scalar
    self.assertTrue(b.d_dt.readonly)        # readonly because of broadcast

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = (1,2,3) + a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a + (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    self.assertEqual(b.shape, b.d_dt.shape)     # d_dt must be broadcasted

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = (1,2,3) + a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a + b
    self.assertEqual(c.d_dt, 6)
    self.assertFalse(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a + b
    self.assertEqual(c.d_dt, 6)
    self.assertTrue(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    # In-place
    a = Scalar((1,2))
    a += 1
    self.assertEqual(a, (2,3))

    a += (2,3)
    self.assertEqual(a, (4,6))
    self.assertTrue(a.is_int())

    self.assertRaises(TypeError, a.__iadd__, 0.5)

    b = Scalar((1,2), mask=(False,True))
    a += b
    self.assertEqual(a[0], 5)
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Scalar((1,2))
    b = Scalar((1,2), derivs={'t':Scalar([(1,1),(2,2)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a += b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, (2,4))
    self.assertEqual(a.d_dt, ((1,1),(2,2)))

    b = Scalar((1,2), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    self.assertRaises(ValueError, a.__iadd__, b)
    self.assertEqual(a, a_copy)

    b = Scalar((1,2), derivs={'t':Scalar(((1,2),(3,4)), drank=1)})
    a += b
    self.assertEqual(a, (3,6))
    self.assertEqual(a.d_dt, ((2,3),(5,6)))

    ############################################################################
    # Subtraction
    ############################################################################

    expr = Scalar(3) - 1
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(3.) - 1
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(3) - 1.
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 3 - Scalar(1)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 3. - Scalar(1)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 3 - Scalar(1.)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((3,4,5)) - 1
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 1 - Scalar((-1,-2,-3))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(1) - (-1,-2,-3)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = (3,4,5) - Scalar(1)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(1) - np.array((-1,-2,-3))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = np.array((3,4,5)) - Scalar(1)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar((3,4,5)) - 1.
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1. - Scalar((-1,-2,-3))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((3.,4.,5.)) - 1
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1 - Scalar((-1.,-2.,-3.))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(1) - (-1.,-2.,-3.)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (3.,4.,5.) - Scalar(1)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(1.) - (-1,-2,-3)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) - Scalar(-1.)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a - (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    a = Scalar(1, derivs={'t':Scalar(-2)})
    b = (1,2,3) - a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a - (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    self.assertEqual(b.shape, b.d_dt.shape)     # d_dt must be broadcasted

    a = Scalar(1, derivs={'t':Scalar(-2)}).as_readonly()
    b = (1,2,3) - a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, 2)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # because of broadcast

    a = Scalar(1, derivs={'t':Scalar(10)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a - b
    self.assertEqual(c.d_dt, 6)
    self.assertFalse(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(10)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a - b
    self.assertEqual(c.d_dt, 6)
    self.assertTrue(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    # In-place
    a = Scalar((3,4))
    a -= 1
    self.assertEqual(a, (2,3))

    a -= (1,2)
    self.assertEqual(a, (1,1))
    self.assertTrue(a.is_int())

    self.assertRaises(TypeError, a.__isub__, 0.5)

    a = Scalar((3,4))
    b = Scalar((1,2), mask=(False,True))
    a -= b
    self.assertEqual(a[0], 2)
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Scalar((2,4))
    b = Scalar((1,2), derivs={'t':Scalar([(1,1),(2,2)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a -= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, (1,2))
    self.assertEqual(a.d_dt, ((-1,-1),(-2,-2)))

    b = Scalar((1,2), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    self.assertRaises(ValueError, a.__isub__, b)
    self.assertEqual(a, a_copy)

    b = Scalar((1,2), derivs={'t':Scalar(((1,2),(3,4)), drank=1)})
    a -= b
    self.assertEqual(a, (0,0))
    self.assertEqual(a.d_dt, ((-2,-3),(-5,-6)))

    ############################################################################
    # Multiplication
    ############################################################################

    expr = Scalar(1) * 2
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(1.) * 2
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(1) * 2.
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1 * Scalar(2)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 1. * Scalar(2)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 1 * Scalar(2.)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((1,2,3)) * 2
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 2 * Scalar((1,2,3))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(2) * (1,2,3)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = (1,2,3) * Scalar(2)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(2) * np.array((1,2,3))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = np.array((1,2,3)) * Scalar(2)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar((1,2,3)) * 2.
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 2. * Scalar((1,2,3))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((1.,2.,3.)) * 2
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 2 * Scalar((1.,2.,3.))
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(2) * (1.,2.,3.)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (1.,2.,3.) * Scalar(2)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(2.) * (1,2,3)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (1,2,3) * Scalar(2.)
    self.assertEqual(expr, (2,4,6))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a * (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (2,4,6))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = (1,2,3) * a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (2,4,6))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a * (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (2,4,6))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = (1,2,3) * a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (2,4,6))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(3)})
    c = a * b
    self.assertEqual(c.d_dt, 7)
    self.assertFalse(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(3)}).as_readonly()
    c = a * b
    self.assertEqual(c.d_dt, 7)
    self.assertTrue(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    # In-place
    a = Scalar((1,2))
    a *= 2
    self.assertEqual(a, (2,4))

    a *= (1,2)
    self.assertEqual(a, (2,8))
    self.assertTrue(a.is_int())

    a = Scalar((1,2))
    self.assertRaises(TypeError, a.__imul__, 0.5)

    a = Scalar((3,4))
    b = Scalar((1,2), mask=(False,True))
    a *= b
    self.assertEqual(a[0], 3)
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Scalar((1,2))
    b = Scalar((3,2), derivs={'t':Scalar([(1,3),(2,1)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a *= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, (3,4))
    self.assertEqual(a.d_dt, ((1,3),(4,2)))

    b = Scalar((2,1), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    self.assertRaises(ValueError, a.__imul__, b)
    self.assertEqual(a, a_copy)

    b = Scalar((2,1), derivs={'t':Scalar(((1,2),(3,4)), drank=1)})
    a *= b
    self.assertEqual(a, (6,4))
    self.assertEqual(a.d_dt, ((5,12),(16,18)))
        # ((3*(1,2) + 2*(1,3), (4*(3,4) + 1*(4,2)

    ############################################################################
    # Division
    ############################################################################

    expr = Scalar(4) / 2
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 4 / Scalar(2)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((2,4,6)) / 2
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 6 / Scalar((6,3,2))
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(6) / (6,3,2)
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = (2,4,6) / Scalar(2)
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(6) / np.array((6,3,2))
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = np.array((2,4,6)) / Scalar(2)
    self.assertEqual(expr, (1,2,3))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(6)})
    b = a / (6,3,2)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (1,2,3))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(2, derivs={'t':Scalar(2)})
    b = (-2,-4,-6) / a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (1,2,3))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(2, derivs={'t':Scalar(2)}).as_readonly()
    b = (-2,-4,-6) / a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, (1,2,3))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar(5, derivs={'t':Scalar(6)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(4)})
    c = a / b
    self.assertEqual(c.d_dt, -2)
    self.assertFalse(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Scalar(5, derivs={'t':Scalar(6)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(4)}).as_readonly()
    c = a / b
    self.assertEqual(c.d_dt, -2)
    self.assertTrue(b.readonly)
    self.assertFalse(c.d_dt.readonly)

    # In-place
    a = Scalar((4,6))
    self.assertRaises(TypeError, a.__itruediv__, 2)

    a = a.as_float()
    a /= 2
    self.assertEqual(a, (2,3))

    a /= (2,1)
    self.assertEqual(a, (1,3))

    a = Scalar((1.,2.))
    a /= 0.5
    self.assertEqual(a, (2,4))

    a = Scalar((3.,4.))
    b = Scalar((1,2), mask=(False,True))
    a /= b
    self.assertEqual(a[0], 3)
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Scalar((12.,15.))
    b = Scalar((3,5), derivs={'t':Scalar([(18,9),(5,-10)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a /= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, (4,3))
    #self.assertEqual(a.d_dt, (-24,-12),(-3,6))
        # (12/(-9)*(18,9), 15/(-25)*(5,-10)) = ((-24,-12),(-3,6))
    self.assertAlmostEqual(a.d_dt.values[0,0], -24, delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[0,1], -12, delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[1,0],  -3, delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[1,1],   6, delta=1.e-14)

    b = Scalar((2,1), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    self.assertRaises(ValueError, a.__imul__, b)
    self.assertEqual(a, a_copy)

    b = Scalar((2,1), derivs={'t':Scalar(((1,1),(1,1)), drank=1)})
    a /= b
    self.assertEqual(a, (2,3))
    #self.assertEqual(a.d_dt, ((-13,-7),(-6,3)))
        # ((1/2)*(-24,-12) + 4/(-4)*(1,1), 1/1*(-3,6) + 3/(-1)*(1,1))
        #       = ((-13,-7),(-6,3)
    self.assertAlmostEqual(a.d_dt.values[0,0], -13, delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[0,1],  -7, delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[1,0],  -6, delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[1,1],   3, delta=1.e-14)

    a /= 2
    self.assertEqual(a, (1,1.5))
    self.assertAlmostEqual(a.d_dt.values[0,0], -13/2., delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[0,1],  -7/2., delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[1,0],  -6/2., delta=1.e-14)
    self.assertAlmostEqual(a.d_dt.values[1,1],   3/2., delta=1.e-14)

    a /= 0
    self.assertTrue(a.mask)

    ############################################################################
    # Floor division
    ############################################################################

    expr = Scalar(5) // 2
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(5.) // 2
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(5) // 2.
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 5 // Scalar(2)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 5. // Scalar(2)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 5 // Scalar(2.)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((5,7,9)) // 2
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar((5.,7.,9.)) // 2
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((5,7,9)) // 2.
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 9 // Scalar((4,3,2))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 9. // Scalar((4,3,2))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 9 // Scalar((4.,3.,2.))
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = np.array((5,7,9)) // Scalar(2)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    # Derivatives, readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a // (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a // (1,2,3)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = (1,2,3) // a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a // b
    self.assertFalse(b.readonly)
    self.assertFalse(c.readonly)

    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a // b
    self.assertFalse(c.readonly)

    # In-place
    a = Scalar((4,6))
    a //= 2
    self.assertEqual(a, (2,3))

    a //= (2,1)
    self.assertEqual(a, (1,3))
    self.assertTrue(a.is_int())

    a = Scalar((1,2))
    self.assertRaises(TypeError, a.__ifloordiv__, 0.5)

    a = Scalar((1.,2.))
    a //= 0.5
    self.assertEqual(a, (2,4))
    self.assertTrue(a.is_float())

    a = Scalar((3,4))
    b = Scalar((1,2), mask=(False,True))
    a //= b
    self.assertEqual(a[0], 3)
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Scalar((12,15))
    b = Scalar((3,5), derivs={'t':Scalar([(18,9),(5,-10)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a //= b
    self.assertFalse(hasattr(a, 'd_dt'))    # no derivatives in floor division

    a = Scalar((12,15))
    a //= 4
    self.assertEqual(a, (3,3))

    a //= 0
    self.assertTrue(a.mask)

    ############################################################################
    # Modulus
    ############################################################################

    expr = Scalar(5) % 3
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar(5.) % 3
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar(5) % 3.
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 5 % Scalar(3)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 5. % Scalar(3)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 5 % Scalar(3.)
    self.assertEqual(expr, 2)
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((7,8,9)) % 5
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = Scalar((7.,8.,9.)) % 5
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = Scalar((7,8,9)) % 5.
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 9 % Scalar((3,4,5))
    self.assertEqual(expr, (0,1,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    expr = 9. % Scalar((3,4,5))
    self.assertEqual(expr, (0,1,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = 9 % Scalar((3.,4.,5.))
    self.assertEqual(expr, (0,1,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_float())

    expr = np.array((7,8,9)) % Scalar(5)
    self.assertEqual(expr, (2,3,4))
    self.assertEqual(type(expr), Scalar)
    self.assertTrue(expr.is_int())

    # Derivatives, readonly
    a = Scalar(9, derivs={'t':Scalar(2)})
    b = a % (3,4,5)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(a.d_dt, b.d_dt)
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar(9, derivs={'t':Scalar(2)}).as_readonly()
    b = a % (3,4,5)
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertTrue(a.d_dt, b.d_dt)
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar(5, derivs={'t':Scalar(2)}).as_readonly()
    b = (7,8,9) % a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)

    a = Scalar(5, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a % b
    self.assertFalse(b.readonly)
    self.assertFalse(c.readonly)

    a = Scalar(5, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a % b
    self.assertFalse(c.readonly)

    # In-place
    a = Scalar((5,7))
    a %= 3
    self.assertEqual(a, (2,1))

    a %= (2,3)
    self.assertEqual(a, (0,1))
    self.assertTrue(a.is_int())

    a = Scalar((9.,12.))
    a %= 3.5
    self.assertEqual(a, (2,1.5))
    self.assertTrue(a.is_float())

    a = Scalar((9,12))
    self.assertRaises(TypeError, a.__imod__, 3.5)

    a = Scalar((3,4))
    b = Scalar((4,2), mask=(False,True))
    a %= b
    self.assertEqual(a[0], 3)
    self.assertEqual(a[0].mask, False)
    self.assertEqual(a[1].mask, True)

    a = Scalar((12,15))
    b = Scalar((3,5), derivs={'t':Scalar([(18,9),(5,-10)], drank=1)})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a %= b
    self.assertFalse(hasattr(a, 'd_dt'))    # no derivatives in modulus

    a = Scalar((12,15))
    a %= 4
    self.assertEqual(a, (0,3))

    a %= 0
    self.assertTrue(a.mask)

    ############################################################################
    # Power
    ############################################################################

    a = Scalar(2)
    b = a**1
    self.assertEqual(b, 2)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar(2)
    b = a**2
    self.assertEqual(b, 4)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar(2)
    b = a**3
    self.assertEqual(b, 8)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar(2.)
    b = a**3
    self.assertEqual(b, 8)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar(2)
    b = a**3.
    self.assertEqual(b, 8)
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar((0,1,2,3,4,5))
    b = a**3
    self.assertEqual(b, (0,1,8,27,64,125))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar((0.,1.,2.,3.,4.,5.))
    b = a**3
    self.assertEqual(b, (0,1,8,27,64,125))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar((-2,-1,0,1,2,3,4,5))
    b = a**3
    self.assertEqual(b, (-8,-1,0,1,8,27,64,125))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_int())

    a = Scalar((-2,-1,0,1,2,3,4,5))
    b = a**3.
    self.assertEqual(b, (-8,-1,0,1,8,27,64,125))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())

    a = Scalar((0,1,4,9,16,25))
    b = a**0.5
    self.assertEqual(b, (0,1,2,3,4,5))
    self.assertEqual(type(b), Scalar)
    self.assertTrue(b.is_float())
    self.assertFalse(b.mask)

    a = Scalar((-4,-1,0,1,4,9,16,25))
    b = a**0.5
    self.assertEqual(b[2:], (0,1,2,3,4,5))
    self.assertEqual(type(b), Scalar)
    self.assertTrue((np.all(b.mask == 2*[True] + 6*[False])))

    a = Scalar((-4,-1,0,1,4,9,16,25))
    b = a**(-0.5)
    self.assertEqual(type(b), Scalar)
    self.assertTrue((np.all(b.mask == 3*[True] + 5*[False])))

    a = Scalar((-2,-1,0,1,2,3,4,5))
    b = a**(-1)
    self.assertTrue((np.all(b.mask == 2*[False] + [True] + 5*[False])))

    for i in range(len(a)):
        if a[i] != 0:
            self.assertAlmostEqual(a[i]*b[i], 1., delta=1.e-14)

    # Derivatives
    a = Scalar(np.arange(20) + 1, derivs={'t':Scalar(np.ones(20))})
    b = a**0
    self.assertEqual(b, 1)
    self.assertEqual(b.d_dt, 0)

    b = a**1
    self.assertEqual(b, a)
    self.assertEqual(b.d_dt, 1)

    b = a**2
    self.assertEqual(b, a*a)
    self.assertEqual(b.d_dt, 2*a)

    b = a**3
    self.assertEqual(b, a*a*a)
    self.assertEqual(b.d_dt, 3*a*a)

    b = a**0.5
    self.assertTrue(abs(b - a.sqrt()).max() < 1.e-14)
    self.assertTrue(abs(b.d_dt - 0.5/a.sqrt()).max() < 1.e-14)

    b = a**(-1)
    self.assertTrue(abs(b*a - 1).max() < 1.e-14)
    self.assertTrue(abs(b.d_dt + b*b).max() < 1.e-14)

    # Read-only status
    self.assertFalse(a.readonly)
    self.assertFalse((a**0).readonly)
    self.assertFalse((a**1).readonly)
    self.assertFalse((a**2).readonly)
    self.assertFalse((a**3).readonly)
    self.assertFalse((a**0.5).readonly)
    self.assertFalse((a**(-0.5)).readonly)
    self.assertFalse((a**(-1)).readonly)

    b = a.as_readonly()
    self.assertTrue(b.readonly)
    self.assertFalse((b**0).readonly)
    self.assertFalse((b**1).readonly)
    self.assertFalse((b**2).readonly)
    self.assertFalse((b**3).readonly)
    self.assertFalse((b**0.5).readonly)
    self.assertFalse((b**(-0.5)).readonly)
    self.assertFalse((b**(-1)).readonly)

    ############################################################################
    # Reciprocal
    ############################################################################

    a = Scalar((1,-1))
    b = a.reciprocal()
    self.assertEqual(b, (1,-1))
    self.assertTrue(type(b), Scalar)
    self.assertTrue(b.is_float())       # automatic conversion to float

    a = Scalar((1,-1,0))
    b = a.reciprocal()
    self.assertEqual(b[:2], (1,-1))
    self.assertTrue(type(b), Scalar)
    self.assertFalse(b.mask[0])
    self.assertFalse(b.mask[1])
    self.assertTrue(b.mask[2])

    a = Scalar((-2,-1,0,1,2), derivs={'t':Scalar((1,1,2,2,2))})
    b = a.reciprocal()
    self.assertEqual(b[:2], (-0.5,-1))
    self.assertEqual(b[3:], (1,0.5))
    self.assertTrue(b[2].mask)
    self.assertTrue(hasattr(b, 'd_dt'))

    DEL = 1.e-13
    self.assertAlmostEqual(b.d_dt[0].values, -0.25, DEL)
    self.assertAlmostEqual(b.d_dt[1].values, -1,    DEL)
    self.assertTrue(b.d_dt[2].mask)
    self.assertAlmostEqual(b.d_dt[3].values, -2,    DEL)
    self.assertAlmostEqual(b.d_dt[4].values, -0.5,  DEL)

    self.assertFalse(b.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar((-2,-1,0,1,2), derivs={'t':Scalar((1,1,2,2,2))}).as_readonly()
    b = a.reciprocal()
    self.assertFalse(b.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Scalar((-2,-1,0,1,2), derivs={'t':Scalar((1,1,2,2,2))})
    b = a.reciprocal(recursive=False)
    self.assertFalse(hasattr(b, 'd_dt'))

    a = Scalar((1,-1))
    b = a.reciprocal(nozeros=True)
    self.assertEqual(b, (1,-1))

    a = Scalar((1,-1,0))
    self.assertRaises(ValueError, a.reciprocal, nozeros=True)

    ############################################################################
    # Comparisons
    ############################################################################

    # Individual values
    self.assertTrue(Scalar(-0.3) <= -0.3)
    self.assertTrue(Scalar(-0.3) >= -0.3)
    self.assertFalse(Scalar(-0.3) < -0.3)
    self.assertFalse(Scalar(-0.3) > -0.3)

    self.assertEqual(type(Scalar(-0.3) <= -0.3), bool)
    self.assertEqual(type(Scalar(-0.3) >= -0.3), bool)
    self.assertEqual(type(Scalar(-0.3) < -0.3), bool)
    self.assertEqual(type(Scalar(-0.3) > -0.3), bool)

    self.assertTrue(Scalar(-0.3) <= -0.2)
    self.assertTrue(Scalar(-0.3) >= -0.4)
    self.assertTrue(Scalar(-0.3) <  -0.2)
    self.assertTrue(Scalar(-0.3) >  -0.4)

    self.assertFalse(Scalar(1,True) <  2)
    self.assertFalse(Scalar(1,True) <= 2)
    self.assertFalse(Scalar(1,True) >  0)
    self.assertFalse(Scalar(1,True) >= 0)

    self.assertFalse(Scalar(1,True) <  Scalar(2,True))
    self.assertTrue (Scalar(1,True) <= Scalar(0,True))
    self.assertFalse(Scalar(1,True) >  Scalar(0,True))
    self.assertTrue (Scalar(1,True) >= Scalar(2,True))

    # Comparisons: Multiple values
    self.assertTrue((Scalar((-0.1,0.,0.1)) <= (-0.1,0.,0.1)).all())
    self.assertTrue((Scalar((-0.1,0.,0.1)) >= (-0.1,0.,0.1)).all())
    self.assertFalse((Scalar((-0.1,0.,0.1)) < (-0.1,0.,0.1)).all())
    self.assertFalse((Scalar((-0.1,0.,0.1)) > (-0.1,0.,0.1)).all())

    self.assertTrue((Scalar((1,2,3)) >= (1,2,3)).all())
    self.assertEqual(type(Scalar((1,2,3)) >= (1,2,3)), Boolean)
    self.assertEqual(type(Scalar((1,2,3)) <= (1,2,3)), Boolean)
    self.assertEqual(type(Scalar((1,2,3)) >  (1,2,3)), Boolean)
    self.assertEqual(type(Scalar((1,2,3)) <  (1,2,3)), Boolean)

    self.assertTrue( (Scalar((1,2,3)) <= (1,2,3)).all())
    self.assertFalse((Scalar((1,2,3)) >  (1,2,3)).all())
    self.assertFalse((Scalar((1,2,3)) <  (1,2,3)).all())
    self.assertTrue( (Scalar((1,2,3)) >= (0,2,3)).all())
    self.assertFalse((Scalar((1,2,3)) >= (2,2,3)).all())

    self.assertTrue((Scalar((1,2),(True,False)) <=
                     Scalar((0,2),(True,False))).all())
    self.assertTrue((Scalar((1,2),(True,False)) >=
                     Scalar((0,2),(True,False))).all())
    self.assertFalse((Scalar((1,2),(True,False)) <
                      Scalar((2,3),(True,False))).all())
    self.assertFalse((Scalar((1,2),(True,False)) >
                      Scalar((0,1),(True,False))).all())

    # Arrays
    N = 100
    x = Scalar(np.random.randn(N))
    y = Scalar(np.random.randn(N))
    for i in range(N):
        if x.values[i] > y.values[i]:
            self.assertTrue(x[i] > y[i])
            self.assertTrue(x[i] >= y[i])
            self.assertFalse(x[i] < y[i])
            self.assertFalse(x[i] <= y[i])
        else:
            self.assertFalse(x[i] > y[i])
            self.assertFalse(x[i] >= y[i])
            self.assertTrue(x[i] < y[i])
            self.assertTrue(x[i] <= y[i])

    for i in range(N-1):
        if np.all(x.values[i:i+2] > y.values[i:i+2]):
            self.assertTrue((x[i:i+2] > y[i:i+2]).all())
            self.assertTrue((x[i:i+2] >= y[i:i+2]).all())
            self.assertFalse((x[i:i+2] < y[i:i+2]).all())
            self.assertFalse((x[i:i+2] <= y[i:i+2]).all())
        elif np.all(x.values[i:i+2] < y.values[i:i+2]):
            self.assertFalse((x[i:i+2] > y[i:i+2]).all())
            self.assertFalse((x[i:i+2] >= y[i:i+2]).all())
            self.assertTrue((x[i:i+2] < y[i:i+2]).all())
            self.assertTrue((x[i:i+2] <= y[i:i+2]).all())
        else:
            self.assertFalse((x[i:i+2] > y[i:i+2]).all())
            self.assertFalse((x[i:i+2] >= y[i:i+2]).all())
            self.assertFalse((x[i:i+2] < y[i:i+2]).all())
            self.assertFalse((x[i:i+2] <= y[i:i+2]).all())

    # Units
    x = Scalar(np.random.randn(10), units=Units.KM)
    y = Scalar(np.random.randn(10), units=Units.CM)
    self.assertTrue((x > y).mask is False)
    self.assertTrue((x < y).mask is False)
    self.assertTrue((x >= y).mask is False)
    self.assertTrue((x <= y).mask is False)

    x = Scalar(np.random.randn(10), units=Units.KM)
    y = Scalar(np.random.randn(10), units=None)
    self.assertTrue((x > y).mask is False)
    self.assertTrue((x < y).mask is False)
    self.assertTrue((x >= y).mask is False)
    self.assertTrue((x <= y).mask is False)

    x = Scalar(np.random.randn(10), units=Units.KM)
    y = Scalar(np.random.randn(10), units=Units.SECONDS)
    self.assertRaises(ValueError, x.__le__, y)
    self.assertRaises(ValueError, x.__ge__, y)
    self.assertRaises(ValueError, x.__lt__, y)
    self.assertRaises(ValueError, x.__gt__, y)

    # Units should be removed
    x = Scalar(np.random.randn(10), units=Units.KM)
    y = Scalar(np.random.randn(10), units=Units.CM)
    self.assertTrue((x > y).units is None)
    self.assertTrue((x < y).units is None)
    self.assertTrue((x >= y).units is None)
    self.assertTrue((x <= y).units is None)

    # Masks
    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -0.2))
    y = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -0.2))
    self.assertTrue((x > y).mask is False)
    self.assertTrue((x < y).mask is False)
    self.assertTrue((x >= y).mask is False)
    self.assertTrue((x <= y).mask is False)

    for i in range(N):
        if not x.mask[i] and not y.mask[i]:
            if x.values[i] > y.values[i]:
                self.assertTrue(x[i] > y[i])
                self.assertTrue(x[i] >= y[i])
                self.assertFalse(x[i] < y[i])
                self.assertFalse(x[i] <= y[i])
            else:
                self.assertFalse(x[i] > y[i])
                self.assertFalse(x[i] >= y[i])
                self.assertTrue(x[i] < y[i])
                self.assertTrue(x[i] <= y[i])
        elif x.mask[i] and y.mask[i]:
            self.assertTrue(x[i] >= y[i])
            self.assertTrue(x[i] <= y[i])
            self.assertFalse(x[i] > y[i])
            self.assertFalse(x[i] < y[i])
        else:
            self.assertFalse(x[i] >= y[i])
            self.assertFalse(x[i] <= y[i])
            self.assertFalse(x[i] > y[i])
            self.assertFalse(x[i] < y[i])

    # Read-only status should be preserved
    x = Scalar(np.random.randn(N))
    y = Scalar(np.random.randn(N))
    self.assertFalse(x.readonly)
    self.assertFalse(y.readonly)
    self.assertTrue(x.as_readonly().readonly)
    self.assertTrue(y.as_readonly().readonly)

    self.assertFalse((x <  y).readonly)
    self.assertFalse((x >  y).readonly)
    self.assertFalse((x <= y).readonly)
    self.assertFalse((x >= y).readonly)

    self.assertFalse((x.as_readonly() <  y).readonly)
    self.assertFalse((x.as_readonly() >  y).readonly)
    self.assertFalse((x.as_readonly() <= y).readonly)
    self.assertFalse((x.as_readonly() >= y).readonly)

    self.assertFalse((x <  y.as_readonly()).readonly)
    self.assertFalse((x >  y.as_readonly()).readonly)
    self.assertFalse((x <= y.as_readonly()).readonly)
    self.assertFalse((x >= y.as_readonly()).readonly)

    self.assertFalse((x.as_readonly() <  y.as_readonly()).readonly)
    self.assertFalse((x.as_readonly() >  y.as_readonly()).readonly)
    self.assertFalse((x.as_readonly() <= y.as_readonly()).readonly)
    self.assertFalse((x.as_readonly() >= y.as_readonly()).readonly)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
