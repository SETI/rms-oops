################################################################################
# Tests for Quaternion to_parts() and from_parts()
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Quaternion, Matrix3, Scalar

class Test_Quaternion_parts(unittest.TestCase):

  def runTest(self):

    a = Quaternion.from_parts(1., [(1,0,0),(0,1,0),(0,0,1)])
    self.assertEqual(a.shape, (3,))
    self.assertEqual(a[0], (1,1,0,0))
    self.assertEqual(a[1], (1,0,1,0))
    self.assertEqual(a[2], (1,0,0,1))

    self.assertTrue(a.readonly)

    a = Quaternion.from_parts(1., [(1,0,0),(0,1,0),(0,0,1)])
    a.insert_deriv('t', Quaternion((1.,2.,3.,4.)))

    self.assertEqual(a.d_dt.shape, (3,))
    self.assertEqual(a.d_dt[0], (1,2,3,4))
    self.assertEqual(a.d_dt[1], (1,2,3,4))
    self.assertEqual(a.d_dt[2], (1,2,3,4))

    angle = Scalar(0., derivs={'t': Scalar(1.)})
    a = Quaternion.from_rotation(angle, [(1,0,0),(0,1,0),(0,0,1)])

    (s,v) = a.to_parts()
    self.assertEqual(s, 1.)
    self.assertEqual(s.d_dt, 0.)

    self.assertEqual(v, (0,0,0))
    self.assertEqual(v[0].d_dt, (0.5,0,0))
    self.assertEqual(v[1].d_dt, (0,0.5,0))
    self.assertEqual(v[2].d_dt, (0,0,0.5))

    self.assertTrue(s.readonly)
    self.assertTrue(v.readonly)

    ####
    N = 100
    q = Quaternion(np.random.randn(N,4), mask=(np.random.rand(N) < 0.2))
    dq_dt = Quaternion(np.random.randn(N,4,2), mask=(np.random.rand(N) < 0.2),
                       drank=1)
    q.insert_deriv('t', dq_dt)

    (s,v) = q.to_parts(recursive=False)
    self.assertEqual(hasattr(q, 'd_dt'), True)
    self.assertEqual(hasattr(s, 'd_dt'), False)
    self.assertEqual(hasattr(v, 'd_dt'), False)
    self.assertEqual(q.readonly, False)
    self.assertEqual(s.readonly, False)
    self.assertEqual(v.readonly, False)

    self.assertTrue(np.all(s.values == q.values[...,0]))
    self.assertTrue(np.all(v.values == q.values[...,1:4]))

    s.values[0] = 42.
    self.assertEqual(q.values[0,0], 42.)    # demonstrates shared memory

    v.values[0,0] = 42.
    self.assertEqual(q.values[0,1], 42.)

    (s,v) = q.to_parts(recursive=True)
    self.assertEqual(hasattr(q, 'd_dt'), True)
    self.assertEqual(hasattr(s, 'd_dt'), True)
    self.assertEqual(hasattr(v, 'd_dt'), True)
    self.assertEqual(q.readonly, False)
    self.assertEqual(s.readonly, False)
    self.assertEqual(v.readonly, False)
    self.assertEqual(q.d_dt.readonly, False)
    self.assertEqual(s.d_dt.readonly, False)
    self.assertEqual(v.d_dt.readonly, False)

    self.assertTrue(np.all(s.d_dt.values == q.d_dt.values[...,0,:]))
    self.assertTrue(np.all(v.d_dt.values == q.d_dt.values[...,1:4,:]))

    q = q.as_readonly()
    (s,v) = q.to_parts(recursive=True)
    self.assertEqual(q.readonly, True)
    self.assertEqual(s.readonly, True)
    self.assertEqual(v.readonly, True)
    self.assertEqual(q.d_dt.readonly, True)
    self.assertEqual(s.d_dt.readonly, True)
    self.assertEqual(v.d_dt.readonly, True)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
