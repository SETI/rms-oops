################################################################################
# Tests for Qube derivative methods
#   insert_deriv(self, key, deriv, override=False)
#   insert_derivs(self, dict, override=False)
#   delete_deriv(self, key)
#   delete_derivs(self)
#   without_derivs(self)
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

#*******************************************************************************
# Test_Qube_derivs
#*******************************************************************************
class Test_Qube_derivs(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    a = Scalar((1,2,3))
    self.assertEqual(a.derivs, {})

    #----------------------------------
    # shape mismatch raises error
    #----------------------------------
    self.assertRaises(ValueError, a.insert_deriv, 't', Scalar((1,2,3,4)))

    #------------------------------------
    # numerator mismatch raises error
    #------------------------------------
    self.assertRaises(ValueError, a.insert_deriv, 't', Vector((1,2,3)))

    #-----------------------------------
    # no derivatives of derivatives
    #-----------------------------------
    a = Scalar((1,2,3))
    b = Scalar((2,3,4))
    c = Scalar((3,4,5))

    b.insert_deriv('t', c)
    a.insert_deriv('t', b)
    self.assertEqual(hasattr(a, 'd_dt'), True)
    self.assertEqual(hasattr(b, 'd_dt'), True)
    self.assertEqual(hasattr(a.d_dt, 'd_dt'), False)

    #----------------------------
    # deleting one derivative
    #----------------------------
    a = Scalar((1,2,3), derivs={'t': Scalar((4,5,6)), 'x': Scalar((5,6,7))})
    self.assertEqual(hasattr(a, 'd_dt'), True)
    self.assertEqual(hasattr(a, 'd_dx'), True)
    a.delete_deriv('t')
    self.assertEqual(hasattr(a, 'd_dt'), False)
    self.assertEqual(hasattr(a, 'd_dx'), True)
    self.assertIn('x', a.derivs)
    self.assertNotIn('t', a.derivs)

    #-----------------------------
    # deleting all derivatives
    #-----------------------------
    a = Scalar((1,2,3), derivs={'t': Scalar((4,5,6)), 'x': Scalar((5,6,7))})
    self.assertEqual(hasattr(a, 'd_dt'), True)
    self.assertEqual(hasattr(a, 'd_dx'), True)
    a.delete_derivs()
    self.assertEqual(hasattr(a, 'd_dt'), False)
    self.assertEqual(hasattr(a, 'd_dx'), False)

    #-----------------------------------
    # changing derivatives, readonly
    #-----------------------------------
    a = Scalar((1,2,3), derivs={'t': Scalar((4,5,6)), 'x': Scalar((5,6,7))})
    self.assertEqual(a.d_dx.readonly, False)

    a = a.as_readonly()

    self.assertEqual(a.d_dt.readonly, True)
    self.assertEqual(a.d_dx.readonly, True)

    self.assertRaises(ValueError, a.delete_deriv, 't')
    self.assertRaises(ValueError, a.delete_derivs)

    self.assertRaises(ValueError, a.insert_derivs, {'a': Scalar((7,8,9)),
                                                    'b': Scalar((8,9,0)),
                                                    'c': Scalar((8,9,0)),
                                                    'd': Scalar((8,9,0)),
                                                    'e': Scalar((8,9,0)),
                                                    'f': Scalar((8,9,0)),
                                                    'g': Scalar((8,9,0)),
                                                    't': Scalar((8,9,0))}) #!!
    self.assertEqual(len(a.derivs), 2)

    a.insert_derivs({'a': Scalar((7,8,9)),
                     'b': Scalar((8,9,0)),
                     'c': Scalar((8,9,0)),
                     'd': Scalar((8,9,0)),
                     'e': Scalar((8,9,0)),
                     'f': Scalar((8,9,0)),
                     'g': Scalar((8,9,0))})

    self.assertEqual(len(a.derivs), 9)

    a.insert_deriv('h', Scalar((7,8,9)))

    self.assertEqual(len(a.derivs), 10)

    self.assertRaises(ValueError, a.insert_derivs, {'a': Scalar((7,8,9))})

    #--------------------
    # without_derivs
    #--------------------
    a = Scalar((1,2,3))
    a.insert_derivs({'a': Scalar((7,8,9)),
                     'b': Scalar((8,9,0)),
                     'c': Scalar((4,5,6)),
                     't': Scalar((5,6,7))})
    self.assertEqual(a.without_derivs().derivs, {})
    self.assertEqual(a.without_derivs(preserve='xxx').derivs, {})
    self.assertEqual(a.without_derivs(preserve=['xxx','yyy']).derivs, {})

    c = a.without_derivs(preserve=['t','xxx'])
    self.assertNotIn('a', c.derivs)
    self.assertNotIn('b', c.derivs)
    self.assertNotIn('c', c.derivs)
    self.assertIn('t', c.derivs)

    self.assertFalse(hasattr(c, 'd_da'))
    self.assertFalse(hasattr(c, 'd_db'))
    self.assertFalse(hasattr(c, 'd_dc'))
    self.assertTrue( hasattr(c, 'd_dt'))

    self.assertFalse(a.readonly)
    self.assertFalse(a.d_da.readonly)
    self.assertFalse(a.d_dt.readonly)

    a = a.as_readonly()

    self.assertTrue(a.readonly)
    self.assertTrue(a.d_da.readonly)
    self.assertTrue(a.d_dt.readonly)

    b = a.without_derivs()

    self.assertTrue(b.readonly)
  #=============================================================================



#*******************************************************************************



############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
