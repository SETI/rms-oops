################################################################################
# tests/fov/test_barrelfov.py
################################################################################

import numpy as np
import time
import unittest

from polymath    import Pair
from oops.config import LOGGING
from oops.fov    import BarrelFOV


class Test_BarrelFOV(unittest.TestCase):

    def runTest(self):

        np.random.seed(8372)

        BarrelFOV.DEBUG = False
        SpeedTest = False

        ########################################
        # Only xy_from_uv defined
        ########################################

        # These are JunoCam parameters
        coefft_xy_from_uv = np.array([1.,
                                      0.,
                                     -5.9624209455667325e-08,
                                      0.,
                                      2.7381910042256151e-14])
        scale = 0.00067540618
        shape = (1648,128)
        fov = BarrelFOV(scale, shape, coefft_xy_from_uv=coefft_xy_from_uv)

        self.assertTrue(fov.max_inversion_error() < 3.e-13)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(0,1648,20), np.arange(0,129,8))
        uv.insert_deriv('t' , Pair(np.random.randn(83,17,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(83,17,2,2), drank=1))

        if SpeedTest:
            iters = 200
            t0 = time.time()
            for k in range(iters):
                xy = fov.xy_from_uv(uv, derivs=True)
                uv_test = fov.uv_from_xy(xy, derivs=True)
            t1 = time.time()
            LOGGING.print('time = %.2f ms' % ((t1-t0)/iters*1000.))
        else:
            xy = fov.xy_from_uv(uv, derivs=True)
            uv_test = fov.uv_from_xy(xy, derivs=False)

        uv_test = fov.uv_from_xy(xy)
        self.assertTrue(abs(uv - uv_test).max() < 3.e-13)

        EPS = 1.e-6
        xy0 = fov.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0]    + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals         - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(83,17,2)))
        xy.insert_deriv('rs', Pair(np.random.randn(83,17,2,2), drank=1))
        uv = fov.uv_from_xy(xy, derivs=True)

        EPS = 1.e-6
        uv0 = fov.uv_from_xy(xy + (-EPS,0), False)
        uv1 = fov.uv_from_xy(xy + ( EPS,0), False)
        duv_dx = (uv1 - uv0) / (2. * EPS)

        uv0 = fov.uv_from_xy(xy + (0,-EPS), False)
        uv1 = fov.uv_from_xy(xy + (0, EPS), False)
        duv_dy = (uv1 - uv0) / (2. * EPS)

        duv_dt = duv_dx * xy.d_dt.vals[...,0]    + duv_dy * xy.d_dt.vals[...,1]
        duv_dr = duv_dx * xy.d_drs.vals[...,0,0] + duv_dy * xy.d_drs.vals[...,1,0]
        duv_ds = duv_dx * xy.d_drs.vals[...,0,1] + duv_dy * xy.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(uv.d_dt.vals         - duv_dt.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,0] - duv_dr.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,1] - duv_ds.vals).max() <= DEL)

        ########################################
        # Only xy_from_uv defined
        ########################################

        coefft_uv_from_xy = np.array([1.000,
                                      0,
                                     -5.9624209455667325e-08,
                                      0,
                                      2.7381910042256151e-14])
        scale = 0.00067540618
        shape = (1648,128)
        fov = BarrelFOV(scale, shape, coefft_uv_from_xy=coefft_uv_from_xy)

        self.assertTrue(fov.max_inversion_error() < 3.e-13)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(20), np.arange(15))
        uv.insert_deriv('t' , Pair(np.random.randn(20,15,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(20,15,2,2), drank=1))
        xy = fov.xy_from_uv(uv, derivs=True)

        uv_test = fov.uv_from_xy(xy)
        self.assertTrue(abs(uv - uv_test).max() < 3.e-13)

        EPS = 1.e-6
        xy0 = fov.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0]    + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals         - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(20,15,2)))
        xy.insert_deriv('rs', Pair(np.random.randn(20,15,2,2), drank=1))
        uv = fov.uv_from_xy(xy, derivs=True)

        xy_test = fov.xy_from_uv(uv)
        self.assertTrue(abs(xy - xy_test).max() < 1.e-14)

        EPS = 1.e-6
        uv0 = fov.uv_from_xy(xy + (-EPS,0), False)
        uv1 = fov.uv_from_xy(xy + ( EPS,0), False)
        duv_dx = (uv1 - uv0) / (2. * EPS)

        uv0 = fov.uv_from_xy(xy + (0,-EPS), False)
        uv1 = fov.uv_from_xy(xy + (0, EPS), False)
        duv_dy = (uv1 - uv0) / (2. * EPS)

        duv_dt = duv_dx * xy.d_dt.vals[...,0]    + duv_dy * xy.d_dt.vals[...,1]
        duv_dr = duv_dx * xy.d_drs.vals[...,0,0] + duv_dy * xy.d_drs.vals[...,1,0]
        duv_ds = duv_dx * xy.d_drs.vals[...,0,1] + duv_dy * xy.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(uv.d_dt.vals         - duv_dt.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,0] - duv_dr.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,1] - duv_ds.vals).max() <= DEL)

        ########################################
        # Both directions
        ########################################

        coefft_xy_from_uv = np.array([1., 6.e-04, -3.e-07, 2.e-10, 7.e-10])

        # From fitting...
        coefft_uv_from_xy = np.array([ 1.00242964e+00,
                                      -6.06115855e-04,
                                      -8.13033063e-07,
                                       5.65874210e-09,
                                      -4.30619511e-10,
                                       1.31761955e-12])

        fov = BarrelFOV(0.001, (100,100),
                        coefft_xy_from_uv=coefft_xy_from_uv,
                        coefft_uv_from_xy=coefft_uv_from_xy, fast=False)

        fov_fast = BarrelFOV(0.001, (100,100),
                             coefft_xy_from_uv=coefft_xy_from_uv,
                             coefft_uv_from_xy=coefft_uv_from_xy, fast=True)

        self.assertTrue(fov.max_inversion_error() < 3.e-14)
        self.assertTrue(fov_fast.max_inversion_error() < 0.3)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(0,101,10), np.arange(0,101,10))
        uv.insert_deriv('t' , Pair(np.random.randn(11,11,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(11,11,2,2), drank=1))

        xy = fov.xy_from_uv(uv, derivs=True)
        uv_test = fov.uv_from_xy(xy, derivs=False)

        uv_test = fov.uv_from_xy(xy)
        self.assertTrue(abs(uv - uv_test).max() < 3.e-13)

        EPS = 1.e-6
        xy0 = fov.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0]    + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 0.02
        self.assertTrue(abs(xy.d_dt.vals         - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(11,11,2)))
        xy.insert_deriv('rs', Pair(np.random.randn(11,11,2,2), drank=1))
        uv = fov.uv_from_xy(xy, derivs=True)

        EPS = 1.e-6
        uv0 = fov.uv_from_xy(xy + (-EPS,0), False)
        uv1 = fov.uv_from_xy(xy + ( EPS,0), False)
        duv_dx = (uv1 - uv0) / (2. * EPS)

        uv0 = fov.uv_from_xy(xy + (0,-EPS), False)
        uv1 = fov.uv_from_xy(xy + (0, EPS), False)
        duv_dy = (uv1 - uv0) / (2. * EPS)

        duv_dt = duv_dx * xy.d_dt.vals[...,0]    + duv_dy * xy.d_dt.vals[...,1]
        duv_dr = duv_dx * xy.d_drs.vals[...,0,0] + duv_dy * xy.d_drs.vals[...,1,0]
        duv_ds = duv_dx * xy.d_drs.vals[...,0,1] + duv_dy * xy.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(uv.d_dt.vals         - duv_dt.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,0] - duv_dr.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,1] - duv_ds.vals).max() <= DEL)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
