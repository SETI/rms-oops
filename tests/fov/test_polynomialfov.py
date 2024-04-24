################################################################################
# tests/fov/test_polynomialfov.py
################################################################################

import numpy as np
import time
import unittest

from polymath    import Pair
from oops.config import LOGGING
from oops.fov    import PolynomialFOV


class Test_PolynomialFOV(unittest.TestCase):

    def runTest(self):

        np.random.seed(5294)

        PolynomialFOV.DEBUG = False
        SpeedTest = False

        ########################################
        # Only xy_from_uv defined
        ########################################

        coefft_xy_from_uv = np.zeros((3,3,2))
        coefft_xy_from_uv[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                             [ 1.20, -0.01,  0.00],
                                             [-0.02,  0.00,  0.00]])
        coefft_xy_from_uv[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                             [-0.20, -0.03,  0.00],
                                             [-0.02,  0.00,  0.00]])

        fov = PolynomialFOV((20,15), coefft_xy_from_uv=coefft_xy_from_uv)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(20), np.arange(15))
        uv.insert_deriv('t' , Pair(np.random.randn(20,15,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(20,15,2,2), drank=1))

        if SpeedTest:
            iters = 100
            t0 = time.time()
            for k in range(iters):
                xy = fov.xy_from_uv(uv, derivs=True)
                uv_test = fov.uv_from_xy(xy, derivs=True)
            t1 = time.time()
            LOGGING.info('time = %.2f ms' % ((t1-t0)/iters*1000.), literal=True)
        else:
            xy = fov.xy_from_uv(uv, derivs=True)
            uv_test = fov.uv_from_xy(xy, derivs=False)

        uv_test = fov.uv_from_xy(xy)
        self.assertTrue(abs(uv - uv_test).max() < 1.e-14)

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

        DEL = 1.e-7
        self.assertTrue(abs(uv.d_dt.vals         - duv_dt.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,0] - duv_dr.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,1] - duv_ds.vals).max() <= DEL)

        ########################################
        # Only uv_from_xy defined
        ########################################

        coefft_uv_from_xy = np.zeros((3,3,2))
        coefft_uv_from_xy[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                             [ 1.20, -0.01,  0.00],
                                             [-0.02,  0.00,  0.00]])
        coefft_uv_from_xy[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                             [-0.20, -0.03,  0.00],
                                             [-0.02,  0.00,  0.00]])

        fov = PolynomialFOV((20,15), coefft_uv_from_xy=coefft_uv_from_xy)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(20), np.arange(15))
        uv.insert_deriv('t' , Pair(np.random.randn(20,15,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(20,15,2,2), drank=1))
        xy = fov.xy_from_uv(uv, derivs=True)

        uv_test = fov.uv_from_xy(xy)
        self.assertTrue(abs(uv - uv_test).max() < 1.e-14)

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

        DEL = 1.e-7
        self.assertTrue(abs(uv.d_dt.vals         - duv_dt.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,0] - duv_dr.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_drs.vals[...,1] - duv_ds.vals).max() <= DEL)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
