################################################################################
# Tests for Qube derivative methods
#   insert_deriv(self, key, deriv, override=False, nocopy='')
#   insert_derivs(self, dict, override=False, nocopy='')
#   delete_deriv(self, key)
#   delete_derivs(self)
#   without_derivs(self)
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_derivs(unittest.TestCase):

  def runTest(self):

    a = Scalar((1,2,3))
    self.assertEqual(a.derivs, {})

    # shape mismatch raises error
    self.assertRaises(ValueError, a.insert_deriv, 't', Scalar((1,2,3,4)))

    # numerator mismatch raises error
    self.assertRaises(ValueError, a.insert_deriv, 't', Vector((1,2,3)))

    # no derivatives of derivatives
    a = Scalar((1,2,3))
    b = Scalar((2,3,4))
    c = Scalar((3,4,5))

    b.insert_deriv('t', c)
    a.insert_deriv('t', b)

    self.assertEquals(hasattr(a, 'd_dt'), True)
    self.assertEquals(hasattr(b, 'd_dt'), True)
    self.assertEquals(hasattr(a.d_dt, 'd_dt'), False)

    # derivative is broadcasted if necessary
    a = Scalar((1,2,3), derivs={'t': Scalar(4)})

    self.assertEquals(a.shape, (3,))
    self.assertEquals(a.d_dt.shape, (3,))
    self.assertEquals(a.d_dt, (4,4,4))

    # deleting one derivative
    a = Scalar((1,2,3), derivs={'t': Scalar(4), 'x': Scalar((5,6,7))})
    self.assertEquals(hasattr(a, 'd_dt'), True)
    self.assertEquals(hasattr(a, 'd_dx'), True)
    a.delete_deriv('t')
    self.assertEquals(hasattr(a, 'd_dt'), False)
    self.assertEquals(hasattr(a, 'd_dx'), True)
    self.assertIn('x', a.derivs)
    self.assertNotIn('t', a.derivs)

    # deleting all derivatives
    a = Scalar((1,2,3), derivs={'t': Scalar(4), 'x': Scalar((5,6,7))})
    self.assertEquals(hasattr(a, 'd_dt'), True)
    self.assertEquals(hasattr(a, 'd_dx'), True)
    a.delete_derivs()
    self.assertEquals(hasattr(a, 'd_dt'), False)
    self.assertEquals(hasattr(a, 'd_dx'), False)

    # changing derivatives, readonly
    a = Scalar((1,2,3), derivs={'t': Scalar(4), 'x': Scalar((5,6,7))})
    self.assertEquals(a.d_dt.readonly, True)    # because of broadcast
    self.assertEquals(a.d_dx.readonly, False)

    a = a.as_readonly()

    self.assertEquals(a.d_dt.readonly, True)
    self.assertEquals(a.d_dx.readonly, True)

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

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
