################################################################################
# Qube.stack() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Vector3, Boolean

class Test_Qube_shrink(unittest.TestCase):

  def runTest(self):

    # corners
    values = np.ones((100,200))
    a = Scalar(values)
    self.assertEqual(a.corners, ((0,0), (100,200)))

    a = Scalar(values, mask=True)
    self.assertEqual(a.corners, ((0,0), (0,0)))

    mask = np.ones((100,200), dtype='bool')
    a = Scalar(values, mask)
    self.assertEqual(a.corners, ((0,0), (0,0)))

    mask.fill(False)
    a = Scalar(values, mask)
    self.assertEqual(a.corners, ((0,0), (100,200)))

    mask[0] = True
    a = Scalar(values, mask)
    self.assertEqual(a.corners, ((1,0), (100,200)))

    mask[:,0] = True
    a = Scalar(values, mask)
    self.assertEqual(a.corners, ((1,1), (100,200)))

    values = np.ones((100,200,3))
    a = Vector3(values)
    self.assertEqual(a.corners, ((0,0), (100,200)))

    a = Vector3(values, mask=True)
    self.assertEqual(a.corners, ((0,0), (0,0)))

    mask = np.ones((100,200), dtype='bool')
    a = Vector3(values, mask)
    self.assertEqual(a.corners, ((0,0), (0,0)))

    mask.fill(False)
    a = Vector3(values, mask)
    self.assertEqual(a.corners, ((0,0), (100,200)))

    mask[0] = True
    a = Vector3(values, mask)
    self.assertEqual(a.corners, ((1,0), (100,200)))

    mask[:,0] = True
    a = Vector3(values, mask)
    self.assertEqual(a.corners, ((1,1), (100,200)))

    # slicer
    values = np.ones((100,200))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True

    a = Scalar(values, mask)
    self.assertEqual(a.slicer, (slice(1, 100, None), slice(1, 200, None)))
    self.assertEqual(a[a.slicer].shape, (99,199))

    a = Scalar(values, mask=True)
    self.assertEqual(a.slicer, (slice(0, 0, None), slice(0, 0, None)))
    self.assertEqual(a[a.slicer].shape, (0,0))

    a = Scalar(values, mask=False)
    self.assertEqual(a.slicer, (slice(0, 100, None), slice(0, 200, None)))
    self.assertEqual(a[a.slicer].shape, (100,200))

    values = np.ones((100,200,3))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True

    a = Vector3(values, mask)
    self.assertEqual(a.slicer, (slice(1, 100, None), slice(1, 200, None)))
    self.assertEqual(a[a.slicer].shape, (99,199))

    a = Vector3(values, mask=True)
    self.assertEqual(a.slicer, (slice(0, 0, None), slice(0, 0, None)))
    self.assertEqual(a[a.slicer].shape, (0,0))

    a = Vector3(values, mask=False)
    self.assertEqual(a.slicer, (slice(0, 100, None), slice(0, 200, None)))
    self.assertEqual(a[a.slicer].shape, (100,200))

    # antimask
    values = np.ones((100,200))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True
    a = Scalar(values, mask)
    self.assertTrue(np.all(a.mask ^ a.antimask))

    a = Scalar(values, False)
    self.assertTrue(np.all(a.mask ^ a.antimask))
    self.assertEqual(a[a.antimask], a)

    a = Scalar(values, True)
    self.assertTrue(np.all(a.mask ^ a.antimask))
    self.assertEqual(a[a.antimask], a[:0])

    values = np.ones((100,200,3))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True
    a = Vector3(values, mask)
    self.assertTrue(np.all(a.mask ^ a.antimask))

    a = Vector3(values, False)
    self.assertTrue(np.all(a.mask ^ a.antimask))
    self.assertEqual(a[a.antimask], a)

    a = Vector3(values, True)
    self.assertTrue(np.all(a.mask ^ a.antimask))
    self.assertEqual(a[a.antimask], a[:0])

    # shrink and unshrink
    values = np.arange(100*200).reshape(100,200)
    a = Scalar(values, mask=(np.random.randn(100,200) < 0))

    b = a.shrink()
    self.assertEqual(a, b)

    antimask = np.zeros((100,200), dtype='bool')
    antimask[0] = True
    b = a.shrink(antimask)
    self.assertEqual(b.shape, (200,))
    self.assertTrue(np.all(b.values == np.arange(200)))

    c = b.unshrink(antimask)
    self.assertEqual(a.shape, c.shape)
    self.assertEqual(a[0], c[0])
    self.assertEqual(a,c)   # because of retained link

    delattr(b, '_Qube__shrink_source_')
    c = b.unshrink(antimask)
    self.assertEqual(a.shape, c.shape)
    self.assertEqual(a[0], c[0])
    self.assertTrue(np.all(c.mask[1:]))

    dist = Scalar(np.arange(-50,50)[:,np.newaxis]**2 +
                  np.arange(-100,100)**2).sqrt()
    mask = (dist > 40)
    a = Scalar(dist, mask)
    self.assertEqual(a.corners, ((10,60),(91,141)))

    b = a.shrink(a.antimask)
    delattr(b, '_Qube__shrink_source_')
    c = b.unshrink(a.antimask)
    self.assertEqual(a, c)

    antimask = a.antimask
    v = Vector3(np.random.randn(100,200,3),
                mask=np.random.randn(100,200) < 0.)
    v2 = v.shrink(antimask)
    v3 = v2.unshrink(antimask)
    self.assertEqual(v, v3)

    v = v.mask_where(~antimask)
    v2 = v.shrink(antimask)
    v3 = v2.unshrink(antimask)
    self.assertEqual(v, v3)

    delattr(v2, '_Qube__shrink_source_')
    v3 = v2.unshrink(antimask)
    self.assertEqual(v, v3)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
