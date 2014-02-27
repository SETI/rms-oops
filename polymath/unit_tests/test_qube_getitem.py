################################################################################
# Qube __getitem__() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_getitem(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # Integers, ellipses, colons, on unmasked objects
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1)

    b = a[0]
    self.assertTrue(np.all(b.values == a.values[0]))
    self.assertTrue(np.all(b.mask == a.mask))
    self.assertEqual(b.shape, (5,6))

    b = a[:,0]
    self.assertTrue(np.all(b.values == a.values[:,0]))
    self.assertTrue(np.all(b.mask == a.mask))
    self.assertEqual(b.shape, (4,6))

    b = a[...,0]
    self.assertTrue(np.all(b.values == a.values[:,:,0]))
    self.assertTrue(np.all(b.mask == a.mask))
    self.assertEqual(b.shape, (4,5))

    ############################################################################
    # Integers, ellipses, colons, on masked objects
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))

    b = a[0]
    self.assertTrue(np.all(b.values == a.values[0]))
    self.assertTrue(np.all(b.mask == a.mask[0]))
    self.assertEqual(b.shape, (5,6))

    b = a[:,0]
    self.assertTrue(np.all(b.values == a.values[:,0]))
    self.assertTrue(np.all(b.mask == a.mask[:,0]))
    self.assertEqual(b.shape, (4,6))

    b = a[...,0]
    self.assertTrue(np.all(b.values == a.values[:,:,0]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,0]))
    self.assertEqual(b.shape, (4,5))

    b = a[0,...,0]
    self.assertTrue(np.all(b.values == a.values[0,:,0]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,0]))
    self.assertEqual(b.shape, (5,))

    self.assertRaises(IndexError, a.__getitem__, (0,0,0,0))

    b = a[...,::-1]
    self.assertTrue(np.all(b.values == a.values[:,:,::-1]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,::-1]))
    self.assertEqual(b.shape, (4,5,6))

    b = a[...,0:5:2]
    self.assertTrue(np.all(b.values == a.values[:,:,0:5:2]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,0:5:2]))
    self.assertEqual(b.shape, (4,5,3))

    ############################################################################
    # Using boolean arrays as masks
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))

    mask = np.array([True,False,False,True])
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (2,5,6))

    mask = np.array([True,False,False,True,True,True])
    b = a[...,mask]
    self.assertTrue(np.all(b.values == a.values[:,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,mask]))
    self.assertEqual(b.shape, (4,5,4))

    mask = np.array([True,False,False,True,True,True])
    b = a[0,...,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask]))

    mask = np.array([True,False,False,True,True,True])
    b = a[0,:,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask]))

    mask = np.array([16*[True] + 4*[False]]).reshape(4,5)
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (16,6))

    mask = np.array([1*[True] + 19*[False]]).reshape(4,5)
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (1,6))

    mask = np.array([1*[True] + 119*[False]]).reshape(4,5,6)
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (1,))

    mask = np.array([1*[True] + 29*[False]]).reshape(5,6)
    b = a[0,mask]
    self.assertTrue(np.all(b.values == a.values[0,mask]))
    self.assertTrue(np.all(b.mask == a.mask[0,mask]))
    self.assertEqual(b.shape, (1,))

    mask = np.array([True,False,False,True])
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (2,5,6))

    mask = np.array([True,False,False,True,True,True])
    b = a[...,mask]
    self.assertTrue(np.all(b.values == a.values[:,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,mask]))
    self.assertEqual(b.shape, (4,5,4))

    mask = np.array([True,False,False,True,True,True])
    b = a[0,...,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask]))

    mask = np.array([True,False,False,True,True,True])
    b = a[0,:,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask]))

    mask = np.array([16*[True] + 4*[False]]).reshape(4,5)
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (16,6))

    mask = np.array([1*[True] + 19*[False]]).reshape(4,5)
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (1,6))

    mask = np.array([1*[True] + 119*[False]]).reshape(4,5,6)
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.shape, (1,))

    mask = np.array([1*[True] + 29*[False]]).reshape(5,6)
    b = a[0,mask]
    self.assertTrue(np.all(b.values == a.values[0,mask]))
    self.assertTrue(np.all(b.mask == a.mask[0,mask]))
    self.assertEqual(b.shape, (1,))

    ############################################################################
    # Using Boolean Qubes as masks
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))

    mask = Boolean(np.array([True,False,False,True]))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (2,5,6))

    mask = Boolean(np.array([True,False,False,True,True,True]))
    b = a[...,mask]
    self.assertTrue(np.all(b.values == a.values[:,:,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,mask.values]))
    self.assertEqual(b.shape, (4,5,4))

    mask = Boolean(np.array([True,False,False,True,True,True]))
    b = a[0,...,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask.values]))

    mask = Boolean(np.array([True,False,False,True,True,True]))
    b = a[0,:,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask.values]))

    mask = Boolean(np.array([16*[True] + 4*[False]]).reshape(4,5))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (16,6))

    mask = Boolean(np.array([1*[True] + 19*[False]]).reshape(4,5))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (1,6))

    mask = Boolean(np.array([1*[True] + 119*[False]]).reshape(4,5,6))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (1,))

    mask = Boolean(np.array([1*[True] + 29*[False]]).reshape(5,6))
    b = a[0,mask]
    self.assertTrue(np.all(b.values == a.values[0,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[0,mask.values]))
    self.assertEqual(b.shape, (1,))

    mask = Boolean(np.array([True,False,False,True]))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (2,5,6))

    mask = Boolean(np.array([True,False,False,True,True,True]))
    b = a[...,mask]
    self.assertTrue(np.all(b.values == a.values[:,:,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,mask.values]))
    self.assertEqual(b.shape, (4,5,4))

    mask = Boolean(np.array([True,False,False,True,True,True]))
    b = a[0,...,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask.values]))

    mask = Boolean(np.array([True,False,False,True,True,True]))
    b = a[0,:,mask]
    self.assertTrue(np.all(b.values == a.values[0,:,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[0,:,mask.values]))

    mask = Boolean(np.array([16*[True] + 4*[False]]).reshape(4,5))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (16,6))

    mask = Boolean(np.array([1*[True] + 19*[False]]).reshape(4,5))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (1,6))

    mask = Boolean(np.array([1*[True] + 119*[False]]).reshape(4,5,6))
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[mask.values]))
    self.assertEqual(b.shape, (1,))

    mask = Boolean(np.array([1*[True] + 29*[False]]).reshape(5,6))
    b = a[0,mask]
    self.assertTrue(np.all(b.values == a.values[0,mask.values]))
    self.assertTrue(np.all(b.mask == a.mask[0,mask.values]))
    self.assertEqual(b.shape, (1,))

    ############################################################################
    # Using tuples, Vectors, Pairs
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))

    tuple = [(0,1,3),(0,1,3)]
    b = a[tuple]
    self.assertTrue(np.all(b.values == a.values[tuple]))
    self.assertTrue(np.all(b.mask == a.mask[tuple]))
    self.assertEqual(b.shape, (3,6))

    pair = Pair([(0,0),(1,1),(3,3)])
    b = a[pair]
    self.assertTrue(np.all(b.values == a.values[pair.as_index()]))
    self.assertTrue(np.all(b.mask == a.mask[pair.as_index()]))
    self.assertEqual(b.shape, (3,6))

    self.assertEqual(a[pair], a[tuple])

    tuple = [(0,1,3),(0,1,3),(0,0,0)]
    b = a[tuple]
    self.assertTrue(np.all(b.values == a.values[tuple]))
    self.assertTrue(np.all(b.mask == a.mask[tuple]))
    self.assertEqual(b.shape, (3,))

    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    b = a[vector]
    self.assertTrue(np.all(b.values == a.values[vector.as_index()]))
    self.assertTrue(np.all(b.mask == a.mask[vector.as_index()]))
    self.assertEqual(b.shape, (3,))

    self.assertEqual(a[vector], a[tuple])

    ############################################################################
    # Read-only status
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    self.assertFalse(a.readonly)

    b = a[0]
    self.assertFalse(b.readonly)

    b = a[:,0]
    self.assertFalse(b.readonly)

    b = a[...,0]
    self.assertFalse(b.readonly)

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2)).as_readonly()
    self.assertTrue(a.readonly)

    b = a[0]
    self.assertTrue(b.readonly)

    b = a[:,0]
    self.assertTrue(b.readonly)

    b = a[...,0]
    self.assertTrue(b.readonly)

    ############################################################################
    # On objects with masks and derivatives
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    da_dt = Vector(np.random.randn(4,5,6,3,2,5), drank=2,
                   mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', da_dt)

    b = a[0]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[0]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[0]))
    self.assertEqual(b.d_dt.shape, (5,6))

    b = a[:,0]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[:,0]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[:,0]))
    self.assertEqual(b.d_dt.shape, (4,6))

    b = a[...,0]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[:,:,0]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[:,:,0]))
    self.assertEqual(b.d_dt.shape, (4,5))

    b = a[0,...,0]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[0,:,0]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[0,:,0]))
    self.assertEqual(b.d_dt.shape, (5,))

    b = a[...,::-1]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[:,:,::-1]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[:,:,::-1]))
    self.assertEqual(b.d_dt.shape, (4,5,6))

    b = a[...,0:5:2]
    self.assertTrue(np.all(b.values == a.values[:,:,0:5:2]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,0:5:2]))
    self.assertEqual(b.d_dt.shape, (4,5,3))

    mask = np.array([True,False,False,True])
    b = a[mask]
    self.assertTrue(np.all(b.values == a.values[mask]))
    self.assertTrue(np.all(b.mask == a.mask[mask]))
    self.assertEqual(b.d_dt.shape, (2,5,6))

    mask = np.array([True,False,False,True,True,True])
    b = a[...,mask]
    self.assertTrue(np.all(b.values == a.values[:,:,mask]))
    self.assertTrue(np.all(b.mask == a.mask[:,:,mask]))
    self.assertEqual(b.d_dt.shape, (4,5,4))

    mask = np.array([True,False,False,True,True,True])
    b = a[0,...,mask]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[0,:,mask]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[0,:,mask]))

    mask = np.array([True,False,False,True,True,True])
    b = a[0,:,mask]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[0,:,mask]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[0,:,mask]))

    mask = np.array([16*[True] + 4*[False]]).reshape(4,5)
    b = a[mask]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[mask]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[mask]))
    self.assertEqual(b.d_dt.shape, (16,6))

    mask = np.array([1*[True] + 19*[False]]).reshape(4,5)
    b = a[mask]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[mask]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[mask]))
    self.assertEqual(b.d_dt.shape, (1,6))

    mask = np.array([1*[True] + 119*[False]]).reshape(4,5,6)
    b = a[mask]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[mask]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[mask]))
    self.assertEqual(b.d_dt.shape, (1,))

    mask = np.array([1*[True] + 29*[False]]).reshape(5,6)
    b = a[0,mask]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[0,mask]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[0,mask]))
    self.assertEqual(b.d_dt.shape, (1,))

    tuple = [(0,1,3),(0,1,3)]
    b = a[tuple]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[tuple]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[tuple]))
    self.assertEqual(b.d_dt.shape, (3,6))

    pair = Pair([(0,0),(1,1),(3,3)])
    b = a[pair]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[pair.as_index()]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[pair.as_index()]))
    self.assertEqual(b.d_dt.shape, (3,6))

    self.assertEqual(a.d_dt[pair], a.d_dt[tuple])

    tuple = [(0,1,3),(0,1,3),(0,0,0)]
    b = a[tuple]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[tuple]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[tuple]))
    self.assertEqual(b.d_dt.shape, (3,))

    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    b = a[vector]
    self.assertTrue(np.all(b.d_dt.values == a.d_dt.values[vector.as_index()]))
    self.assertTrue(np.all(b.d_dt.mask == a.d_dt.mask[vector.as_index()]))
    self.assertEqual(b.d_dt.shape, (3,))

    self.assertEqual(a.d_dt[vector], a.d_dt[tuple])

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
