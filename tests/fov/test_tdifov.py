################################################################################
# tests/fov/test_tdifov.py
################################################################################

import unittest
import numpy as np

from polymath import Scalar, Pair
from oops.fov import TDIFOV


class Test_TDIFOV(unittest.TestCase):

    def runTest(self):

        np.random.seed(9816)
        from oops.fov.flatfov import FlatFOV

        ################################################
        # 10 lines, TDI -v, 8 sec/shift, tstop=100
        ################################################

        staticfov = FlatFOV((1/2048.,-1/2048.), (100,10))
        fov = TDIFOV(staticfov, 100, 8., '-v')

        uv = Pair.combos(np.arange(0,101,50), np.arange(11))
        xy0 = staticfov.xy_from_uvt(uv)

        self.assertEqual(fov.xy_from_uvt(uv, time=100), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=92), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=84)[:,:-1], xy0[:,1:])
        self.assertEqual(fov.xy_from_uvt(uv, time=83)[:,:-2], xy0[:,2:])
        self.assertEqual(fov.xy_from_uvt(uv, time=101)[:,1:], xy0[:,:-1])

        self.assertEqual(fov.uv_from_xyt(xy0, time=100), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=92), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=84)[:,1:], uv[:,:-1])
        self.assertEqual(fov.uv_from_xyt(xy0, time=83)[:,2:], uv[:,:-2])
        self.assertEqual(fov.uv_from_xyt(xy0, time=101)[:,:-1], uv[:,1:])

        # with derivs
        N = 100
        uv = Pair.combos(50 + 20. * np.random.randn(N),
                          5 +  3. * np.random.randn(N))
        time = Scalar(90 + 20 * np.random.randn(N))
        uv.insert_deriv('rs', Pair(np.random.randn(N,2,2), drank=1))
        uv.insert_deriv('q' , Pair(np.random.randn(N,2)))
        uv.insert_deriv('t' , Pair(np.random.randn(N,2)))

        xy0 = staticfov.xy_from_uvt(uv, derivs=True)
        xy  = fov.xy_from_uvt(uv, time=time, derivs=True)
        self.assertEqual(xy0.d_drs, xy.d_drs)
        self.assertEqual(xy0.d_dq,  xy.d_dq)

        diffs = xy0.d_dt - xy.d_dt
        self.assertTrue(np.all(diffs.vals[...,0] == 0))
        self.assertTrue(np.all(abs(diffs.vals[...,1] - 1/2048./8.) < 1.e-14))

        ################################################
        # 10 lines, TDI +v, 8 sec/shift, tstop=100
        ################################################

        staticfov = FlatFOV((1/2048.,-1/2048.), (100,10))
        fov = TDIFOV(staticfov, 100, 8., '+v')

        uv = Pair.combos(np.arange(0,101,50), np.arange(11))
        xy0 = staticfov.xy_from_uvt(uv)

        self.assertEqual(fov.xy_from_uvt(uv, time=100), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=92), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=84)[:,1:], xy0[:,:-1])
        self.assertEqual(fov.xy_from_uvt(uv, time=83)[:,2:], xy0[:,:-2])
        self.assertEqual(fov.xy_from_uvt(uv, time=101)[:,:-1], xy0[:,1:])

        self.assertEqual(fov.uv_from_xyt(xy0, time=100), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=92), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=84)[:,:-1], uv[:,1:])
        self.assertEqual(fov.uv_from_xyt(xy0, time=83)[:,:-2], uv[:,2:])
        self.assertEqual(fov.uv_from_xyt(xy0, time=101)[:,1:], uv[:,:-1])

        # with derivs
        N = 100
        uv = Pair.combos(50 + 20. * np.random.randn(N),
                          5 +  3. * np.random.randn(N))
        time = Scalar(90 + 20 * np.random.randn(N))
        uv.insert_deriv('rs', Pair(np.random.randn(N,2,2), drank=1))
        uv.insert_deriv('q' , Pair(np.random.randn(N,2)))
        uv.insert_deriv('t' , Pair(np.random.randn(N,2)))

        xy0 = staticfov.xy_from_uvt(uv, derivs=True)
        xy  = fov.xy_from_uvt(uv, time=time, derivs=True)
        self.assertEqual(xy0.d_drs, xy.d_drs)
        self.assertEqual(xy0.d_dq,  xy.d_dq)

        diffs = xy0.d_dt - xy.d_dt
        self.assertTrue(np.all(diffs.vals[...,0] == 0))
        self.assertTrue(np.all(abs(diffs.vals[...,1] + 1/2048./8.) < 1.e-14))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
