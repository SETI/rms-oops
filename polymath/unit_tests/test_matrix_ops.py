################################################################################
# Matrix tests for arithmetic operations
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Matrix, Vector, Scalar, Boolean, Units

#*******************************************************************************
# Test_Matrix_ops
#*******************************************************************************
class Test_Matrix_ops(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-------------------------
    # Unary plus
    #-------------------------

    a = Matrix([(1,2,3),(3,4,5)])
    b = +a
    self.assertEqual(b, [(1,2,3),(3,4,5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())       # Matrix is always float
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))

    # Derivatives, readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])})
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,0),(1,1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])}).as_readonly()
    b = +a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,0),(1,1)])
    self.assertTrue(a.readonly)
    self.assertTrue(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__iadd__, [(1,0),(1,1)]) # because readonly

    #-------------------------
    # Unary minus
    #-------------------------

    a = Matrix([(1,2,3),(3,4,5)])
    b = -a
    self.assertEqual(b, [(-1,-2,-3),(-3,-4,-5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())       # Matrix is always float
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertFalse(hasattr(b, 'd_dt'))

    # Derivatives, readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])})
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(-1,-0),(-1,-1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    # Derivatives, readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])}).as_readonly()
    b = -a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(-1,-0),(-1,-1)])
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    self.assertRaises(ValueError, a.__isub__, [(1,0),(1,1)]) # because readonly

    #-------------------------
    # abs()
    #-------------------------

    a = Matrix([(1,0,0),(0,0,1),(0,-1,0)])
    self.assertRaises(TypeError, a.__abs__)

    #--------------------------
    # Addition
    #--------------------------

    a = Matrix([(1,2,3),(3,4,5)])
    b = a + [(1,1,1),(0,0,0)]
    self.assertEqual(b, [(2,3,4),(3,4,5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())       # Matrix is always float

    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1,1),(0,0,0)] + a
    self.assertEqual(b, [(2,3,4),(3,4,5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())

    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1),(0,0)]
    self.assertRaises(ValueError, a.__add__, b)

    # Derivatives, readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = a + [(1,1),(0,0)]
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,1),(-1,-1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = [(1,1),(0,0)] + a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,1),(-1,-1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    a = a.as_readonly()
    b = a + [(1,1),(0,0)]
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,1),(-1,-1)])
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)    # deriv is a direct copy

    # In-place
    a = Matrix([(1,2),(3,4)])
    a += [(1,1),(0,0)]
    self.assertEqual(a, [(2,3),(3,4)])

    a = Matrix([(1,2),(3,4)])
    b = Matrix([[(1,1),(0,0)],[(0,1),(2,0)]])
    self.assertRaises(ValueError, a.__iadd__, b)    # shape mismatch

    a = Matrix([(1,2),(3,4)])
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(1,1),(2,2)])})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a += b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, [(2,3),(3,4)])
    self.assertEqual(a.d_dt, [(1,1),(2,2)])

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,2),(3,4)])})
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(4,3),(2,1)])})
    a += b
    self.assertEqual(a, [(2,3),(3,4)])
    self.assertEqual(a.d_dt, ((5,5),(5,5)))

    #----------------------------
    # Subtraction
    #----------------------------

    a = Matrix([(1,2,3),(3,4,5)])
    b = a - [(1,1,1),(0,0,0)]
    self.assertEqual(b, [(0,1,2),(3,4,5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())       # Matrix is always float

    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1,1),(0,0,0)] - a
    self.assertEqual(b, [(0,-1,-2),(-3,-4,-5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())

    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1),(0,0)]
    self.assertRaises(ValueError, a.__sub__, b)

    # Derivatives, readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = a - [(1,1),(0,0)]
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,1),(-1,-1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = [(1,1),(0,0)] - a
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(-1,-1),(1,1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    a = a.as_readonly()
    b = a - [(1,1),(0,0)]
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(1,1),(-1,-1)])
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertTrue(b.d_dt.readonly)        # deriv is an exact copy

    # In-place
    a = Matrix([(1,2),(3,4)])
    a -= [(1,1),(0,0)]
    self.assertEqual(a, [(0,1),(3,4)])

    a = Matrix([(1,2),(3,4)])
    b = Matrix([[(1,1),(0,0)],[(0,1),(2,0)]])
    self.assertRaises(ValueError, a.__isub__, b)    # shape mismatch

    a = Matrix([(1,2),(3,4)])
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(1,1),(2,2)])})
    self.assertFalse(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))

    a -= b
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertEqual(a, [(0,1),(3,4)])
    self.assertEqual(a.d_dt, [(-1,-1),(-2,-2)])

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,2),(3,4)])})
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(4,3),(2,1)])})
    a -= b
    self.assertEqual(a, [(0,1),(3,4)])
    self.assertEqual(a.d_dt, ((-3,-1),(1,3)))

    #------------------------------
    # Multiplication
    #------------------------------

    a = Matrix([(1,2,3),(3,4,5)])
    b = a * 2
    self.assertEqual(b, [(2,4,6),(6,8,10)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())       # Matrix is always float

    a = Matrix([(1,2,3),(3,4,5)])
    b = 2 * a
    self.assertEqual(b, [(2,4,6),(6,8,10)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())

    a = Matrix([(1,0),(0,1)])
    b = Matrix([(1,2),(3,4)]) * a
    self.assertEqual(b, [(1,2),(3,4)])
    self.assertEqual(type(b), Matrix)

    a = Matrix([(1,0),(0,1)])
    b = a * Matrix([(1,2),(3,4)])
    self.assertEqual(b, [(1,2),(3,4)])
    self.assertEqual(type(b), Matrix)

    a = Matrix([(1,0,-1),(0,2,-1)])
    b = a * Vector((1,2,3))
    self.assertEqual(b, (-2,1))
    self.assertEqual(type(b), Vector)

    a = Matrix([(1,0,-1),(0,2,-1)])
    b = a * Vector([(1,6),(2,5),(3,4)], drank=1)
    self.assertEqual(b, [(-2,2),(1,6)])
    self.assertEqual(type(b), Vector)

    # Derivatives, readonly
    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(3,2,1),(1,1,1)])})
    b = a * 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(6,4,2),(2,2,2)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(3,2,1),(1,1,1)])})
    b = Scalar(2, derivs={'t':Scalar(1)})
    c = a * b
    self.assertEqual(c.d_dt, [(7,4,1),(2,4,1)])

    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(3,2,1),(1,1,1)])})
    a = a.as_readonly()
    b = a * 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(6,4,2),(2,2,2)])
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,0,-1),(0,2,-1)])
    b = Vector((1,2,3))
    c = a * b
    self.assertEqual(c, (-2,1))
    self.assertFalse(c.readonly)

    a = Matrix([(1,0,-1),(0,2,-1)]).as_readonly()
    b = Vector((1,2,3))
    c = a * b
    self.assertEqual(c, (-2,1))
    self.assertFalse(c.readonly)

    a = Matrix([(1,0,-1),(0,2,-1)])
    b = Vector((1,2,3)).as_readonly()
    c = a * b
    self.assertEqual(c, (-2,1))
    self.assertFalse(c.readonly)

    a = Matrix([(1,0,-1),(0,2,-1)]).as_readonly()
    b = Vector((1,2,3)).as_readonly()
    c = a * b
    self.assertEqual(c, (-2,1))
    self.assertFalse(c.readonly)

    # In-place
    a = Matrix([(1,2),(3,4)])
    a *= 2
    self.assertEqual(a, [(2,4),(6,8)])

    a = Matrix([(1,2),(3,4)])
    a *= Matrix([(2,0),(0,2)])
    self.assertEqual(a, [(2,4),(6,8)])

    a = Matrix([(1,2),(3,4)])
    a *= Scalar(2, derivs={'t':Scalar(-1)})
    self.assertEqual(a, [(2,4),(6,8)])
    self.assertEqual(a.d_dt, [(-1,-2),(-3,-4)])

    a = Matrix([(1,2),(3,4)])
    a *= Matrix([(2,0),(0,2)], derivs={'t':Matrix([(-1,0),(0,-1)])})
    self.assertEqual(a, [(2,4),(6,8)])
    self.assertEqual(a.d_dt, [(-1,-2),(-3,-4)])

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(0,1)])})
    a *= Matrix([(2,0),(0,2)], derivs={'t':Matrix([(-1,0),(0,-1)])})
    self.assertEqual(a, [(2,4),(6,8)])
    self.assertEqual(a.d_dt, [(1,-2),(-3,-2)])

    #--------------------------------
    # Division
    #--------------------------------

    a = Matrix([(2,4,6),(6,8,10)])
    b = a / 2
    self.assertEqual(b, [(1,2,3),(3,4,5)])
    self.assertEqual(type(b), Matrix)
    self.assertTrue(b.is_float())       # Matrix is always float

    a = Matrix([(1,2,3),(3,4,5)])
    # b = 2 / a
    self.assertRaises(ValueError, Scalar(2).__div__, a)

    a = Matrix([(1,0),(0,-1)])
    b = 2 / a                           # 2 * inverse matrix
    self.assertEqual(b, [(2,0),(0,-2)])
    self.assertEqual(type(b), Matrix)

    a = Matrix([(-1,0),(0,-1)])
    b = Matrix([(1,2),(3,4)]) / a
    self.assertEqual(b, [(-1,-2),(-3,-4)])
    self.assertEqual(type(b), Matrix)

    a = Matrix([(1,0),(0,-1)])
    b = Matrix([(1,2),(3,4)]) / a
    self.assertEqual(b, [(1,-2),(3,-4)])
    self.assertEqual(type(b), Matrix)

    a = Matrix([(1,2),(3,4)])
    b = Matrix([(1,0),(0,1)]) / a
    self.assertEqual(b, a.reciprocal())
    self.assertEqual(type(b), Matrix)

    a = Matrix([(1,2),(3,4)])
    b = 1. / a
    self.assertEqual(b, a.reciprocal())
    self.assertEqual(type(b), Matrix)

    # Derivatives, readonly
    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(6,4,2),(2,2,2)])})
    b = a / 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(3,2,1),(1,1,1)])
    self.assertFalse(a.readonly)
    self.assertFalse(b.readonly)
    self.assertFalse(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(6,4,2),(2,2,2)])})
    a = a.as_readonly()
    b = a / 2
    self.assertTrue(hasattr(a, 'd_dt'))
    self.assertTrue(hasattr(b, 'd_dt'))
    self.assertEqual(b.d_dt, [(3,2,1),(1,1,1)])
    self.assertTrue(a.readonly)
    self.assertFalse(b.readonly)
    self.assertTrue(a.d_dt.readonly)
    self.assertFalse(b.d_dt.readonly)

    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])})
    b = Scalar(2, derivs={'t':Scalar(1)})
    c = a / b
    self.assertEqual(c.d_dt, -a/b/b*b.d_dt + a.d_dt/b)

    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(1)})
    c = a / b
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])})
    b = Scalar(2, derivs={'t':Scalar(1)}).as_readonly()
    c = a / b
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(1)}).as_readonly()
    c = a / b
    self.assertFalse(c.readonly)
    self.assertFalse(c.d_dt.readonly)

    # In-place
    a = Matrix([(2,4),(6,8)])
    a /= 2
    self.assertEqual(a, [(1,2),(3,4)])

    a = Matrix([(2,4),(6,8)])
    a /= Matrix([(2,0),(0,2)])
    self.assertEqual(a, [(1,2),(3,4)])

    a = Matrix([(2,4),(6,8)])
    b = Scalar(2, derivs={'t':Scalar(-1)})
    da_dt = -a/b/b*b.d_dt
    a /= b
    self.assertEqual(a, [(1,2),(3,4)])
    self.assertEqual(a.d_dt, da_dt)

    a = Matrix([(2,4),(6,8)], derivs={'t':Matrix([(6,4),(2,2)])})
    b = Matrix([(2,0),(0,2)], derivs={'t':Matrix([(-1,0),(0,-1)])})
    da_dt = -a/b.wod/b.wod*b.d_dt + a.d_dt/b.wod
    a /= b
    self.assertEqual(a, [(1,2),(3,4)])
    self.assertEqual(a.d_dt, da_dt)

    #-------------------------------
    # Floor division
    #-------------------------------

    self.assertRaises(TypeError, Matrix([(2,4),(6,8)]).__floordiv__, 1)
    self.assertRaises(TypeError, Matrix([(2,4),(6,8)]).__ifloordiv__, 1)

    #--------------------------------
    # Modulus
    #--------------------------------

    self.assertRaises(TypeError, Matrix([(2,4),(6,8)]).__mod__, 1)
    self.assertRaises(TypeError, Matrix([(2,4),(6,8)]).__imod__, 1)
  #=============================================================================



#*******************************************************************************



################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
