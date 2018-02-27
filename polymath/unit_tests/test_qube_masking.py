################################################################################
# Tests for Qube mask_where() methods
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_qube_masking(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # mask_where()
    ############################################################################

    a = Scalar(np.arange(20))
    b = a.mask_where(10*[True] + 10*[False])
    self.assertEqual(b.sum(), np.sum(np.arange(10,20)))

    a = Scalar(np.arange(20))
    b = a.mask_where(Boolean(10*[True] + 10*[False]))
    self.assertEqual(b.sum(), np.sum(np.arange(10,20)))

    a = Scalar(np.arange(20))
    b = a.mask_where(a % 2 == 1)
    self.assertEqual(b.sum(), 2 * np.sum(np.arange(10)))

    a = Scalar(np.arange(20))
    b = a.mask_where(a % 2 == 1, replace=0, remask=False)
    self.assertEqual(b.sum(), 2 * np.sum(np.arange(10)))

    a = Scalar(np.arange(20))
    b = a.mask_where(a % 2 == 1, replace=10, remask=False)
    self.assertEqual(b.sum(), 100 + 2 * np.sum(np.arange(10)))

    a = Vector(np.ones(60).reshape(20,3)) * np.arange(20)
    b = a.mask_where(10*[True] + 10*[False])
    c = b.to_scalars()
    self.assertTrue(np.all(c[0].mask[0:10] == True))
    self.assertTrue(np.all(c[0].mask[10:20] == False))
    self.assertEqual(c[0].sum(), np.sum(np.arange(10,20)))
    self.assertEqual(c[0], c[1])
    self.assertEqual(c[0], c[2])

    a = Vector(np.ones(60).reshape(20,3)) * np.arange(20)
    b = a.mask_where(20*[True], (1,2,3), remask=False)
    self.assertEqual(b, (1,2,3))
    self.assertEqual(type(b), Vector)

    a = Vector(np.ones(60).reshape(20,3)) * np.arange(20)
    self.assertRaises(ValueError, a.mask_where, 20*[True], (1,2,3,4))

    a = Scalar(np.arange(10))
    b = -a
    c = a.mask_where(a < 5, replace=b, remask=False)
    self.assertEqual(c, [0,-1,-2,-3,-4,5,6,7,8,9])

    v = Vector3(np.arange(12).reshape(4,3))
    c = v.mask_where([1,0,0,0], replace=-v, remask=False)
    self.assertEqual(c, [[0,-1,-2],[3,4,5],[6,7,8],[9,10,11]])

    c = v.mask_where([1,0,0,0], replace=-v, remask=True)
    self.assertEqual(c[0], Vector3.MASKED)
    self.assertEqual(c[1:], [[3,4,5],[6,7,8],[9,10,11]])

    ############################################################################
    # mask_where_eq()
    ############################################################################

    a = Scalar((1,2,3))
    b = a.mask_where_eq(2)
    self.assertEqual(b[0], 1)
    self.assertEqual(b[1].mask, True)
    self.assertEqual(b[2], 3)

    a = Scalar((1,2,3))
    b = a.mask_where_eq(2, 7, remask=False)
    self.assertEqual(b, (1,7,3))

    a = Vector(np.arange(30).reshape(10,3) % 6)
    b = a.mask_where_eq((3,4,5), (0,1,2), remask=False)
    self.assertEqual(b, (0,1,2))

    a = Vector(np.arange(30).reshape(10,3) % 6)
    b = a.mask_where_eq((3,4,5))
    self.assertEqual(np.sum(b.mask), 5)
    self.assertEqual(b[~b.mask], (0,1,2))

    a = Vector(np.arange(30).reshape(10,3) % 6)
    b = a.mask_where_eq((3,4,5), (0,1,2), remask=False)
    self.assertEqual(b.masked(), 0)
    self.assertEqual(b, (0,1,2))

    ############################################################################
    # mask_where_ne()
    ############################################################################

    a = Scalar((1,2,3))
    b = a.mask_where_ne(2)
    self.assertEqual(b[0].mask, True)
    self.assertEqual(b[1], 2)
    self.assertEqual(b[2].mask, True)

    a = Scalar((1,2,3))
    b = a.mask_where_ne(2, 7, remask=False)
    self.assertEqual(b, (7,2,7))

    a = Vector(np.arange(30).reshape(10,3) % 6)
    b = a.mask_where_ne((3,4,5), (3,4,5), remask=False)
    self.assertEqual(b, (3,4,5))
    self.assertEqual(b.masked(), 0)

    a = Vector(np.arange(30).reshape(10,3) % 6)
    b = a.mask_where_eq((3,4,5))
    self.assertEqual(b.masked(), 5)
    self.assertEqual(b[~b.mask], (0,1,2))

    a = Vector(np.arange(30).reshape(10,3) % 6)
    b = a.mask_where_eq((3,4,5), (0,1,2), remask=False)
    self.assertEqual(b.masked(), 0)
    self.assertEqual(b.unmasked(), 10)
    self.assertEqual(b, (0,1,2))

    ############################################################################
    # mask_where_le(), etc.
    ############################################################################

    a = Scalar((1,2,4))
    self.assertEqual(a.mask_where_le(2).masked(), 2)
    self.assertEqual(a.mask_where_lt(2).masked(), 1)
    self.assertEqual(a.mask_where_ge(2).masked(), 2)
    self.assertEqual(a.mask_where_gt(2).masked(), 1)

    self.assertEqual(a.mask_where_le(2).sum(), 4)
    self.assertEqual(a.mask_where_lt(2).sum(), 6)
    self.assertEqual(a.mask_where_ge(2).sum(), 1)
    self.assertEqual(a.mask_where_gt(2).sum(), 3)

    self.assertEqual(a.mask_where_le(2,0,False).masked(), 0)
    self.assertEqual(a.mask_where_lt(2,0,False).masked(), 0)
    self.assertEqual(a.mask_where_ge(2,0,False).masked(), 0)
    self.assertEqual(a.mask_where_gt(2,0,False).masked(), 0)

    self.assertEqual(a.mask_where_le(2,0,False).unmasked(), 3)
    self.assertEqual(a.mask_where_lt(2,0,False).unmasked(), 3)
    self.assertEqual(a.mask_where_ge(2,0,False).unmasked(), 3)
    self.assertEqual(a.mask_where_gt(2,0,False).unmasked(), 3)

    self.assertEqual(a.mask_where_le(2,0,False).sum(), 4)
    self.assertEqual(a.mask_where_lt(2,0,False).sum(), 6)
    self.assertEqual(a.mask_where_ge(2,0,False).sum(), 1)
    self.assertEqual(a.mask_where_gt(2,0,False).sum(), 3)

    ############################################################################
    # mask_where_between(), mask_where_outside()
    ############################################################################

    a = Scalar((1,2,3,4,5,6))
    self.assertEqual(a.mask_where_between(2,4,True, 0,False), (1,0,0,0,5,6))
    self.assertEqual(a.mask_where_between(2,4,False,0,False), (1,2,0,4,5,6))
    self.assertEqual(a.mask_where_outside(2,4,True, 0,False), (0,0,3,0,0,0))
    self.assertEqual(a.mask_where_outside(2,4,False,0,False), (0,2,3,4,0,0))

    self.assertEqual(a.mask_where_between(2,4,True, 0,True).masked(), 3)
    self.assertEqual(a.mask_where_between(2,4,False,0,True).masked(), 1)
    self.assertEqual(a.mask_where_outside(2,4,True, 0,True).masked(), 5)
    self.assertEqual(a.mask_where_outside(2,4,False,0,True).masked(), 3)

    ############################################################################
    # clip()
    ############################################################################

    a = Scalar((1,2,3,4,5,6))
    self.assertEqual(a.clip(2,4,False), (2,2,3,4,4,4))
    self.assertEqual(a.clip(2,4,True).masked(), 3)

    self.assertEqual(a.clip(6*[2],6*[4],False), (2,2,3,4,4,4))
    self.assertEqual(a.clip(None,6*[4],False), (1,2,3,4,4,4))
    self.assertEqual(a.clip(6*[2],6*[4],True).masked(), 3)
    self.assertEqual(a.clip(None,6*[4],True).masked(), 2)

    self.assertEqual(a.clip([7,6,5,4,3,2],[8,7,6,5,4,3],False), (7,6,5,4,4,3))

    upper = Scalar([8,7,6,5,4,3], 5*[False] + [True])
    self.assertEqual(a.clip([7,6,5,4,3,2],upper,False), (7,6,5,4,4,3))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
