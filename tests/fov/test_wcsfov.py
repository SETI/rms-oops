################################################################################
# tests/fov/test_wcsfov.py
################################################################################

import numpy as np
import time
import unittest

from astropy.wcs import WCS

from polymath               import Pair
from oops.fov.polynomialfov import PolynomialFOV
from oops.fov.wcsfov        import WCSFOV
from oops.config            import LOGGING
from oops.constants         import DPR


# From jw01373012001_03103_00003_nrcb4_cal.fits
header1 = {
    'NAXIS1'  :                 2048,
    'NAXIS2'  :                 2048,
    'CRPIX1'  :               1024.5, # axis 1 coordinate of the reference pixel
    'CRPIX2'  :               1024.5, # axis 2 coordinate of the reference pixel
    'CRVAL1'  :    8.404712747544201, # first axis value at the reference pixel
    'CRVAL2'  :      2.1104976237708, # second axis value at the reference pixel
    'CTYPE1'  : 'RA---TAN-SIP'      , # first axis coordinate type
    'CTYPE2'  : 'DEC--TAN-SIP'      , # second axis coordinate type
    'CUNIT1'  : 'deg'               , # first axis units
    'CUNIT2'  : 'deg'               , # second axis units
    'CD1_1'   :  2.6929364608365E-06, # linear transformation matrix element, CD matrix
    'CD1_2'   : -8.2844293112841E-06, # linear transformation matrix element, CD matrix
    'CD2_1'   : -8.2497328917804E-06, # linear transformation matrix element, CD matrix
    'CD2_2'   : -2.7446109068341E-06, # linear transformation matrix element, CD matrix
    'RA_REF'  :    8.404740083800295, # [deg] Right ascension of the reference point
    'DEC_REF' :    2.110432946199729, # [deg] Declination of the reference point
    'A_ORDER' :                    3, # Degree of forward SIP polynomial
    'B_ORDER' :                    3, # Degree of forward SIP polynomial
    'AP_ORDER':                    3, # Degree of inverse SIP polynomial
    'BP_ORDER':                    3, # Degree of inverse SIP polynomial
    'A_0_2'   : 4.03662975957421E-07, # SIP coefficient, forward transform
    'A_0_3'   : -9.8145417884914E-12, # SIP coefficient, forward transform
    'A_1_1'   : -6.8141359259007E-06, # SIP coefficient, forward transform
    'A_1_2'   : 3.87581460247605E-10, # SIP coefficient, forward transform
    'A_2_0'   : -2.0860219723753E-06, # SIP coefficient, forward transform
    'A_2_1'   : 4.59927104696853E-11, # SIP coefficient, forward transform
    'A_3_0'   : 3.27716616827621E-10, # SIP coefficient, forward transform
    'B_0_2'   : -4.7832813609093E-06, # SIP coefficient, forward transform
    'B_0_3'   : 3.49102783601233E-10, # SIP coefficient, forward transform
    'B_1_1'   : -2.5703378553347E-06, # SIP coefficient, forward transform
    'B_1_2'   : 1.43658260461187E-11, # SIP coefficient, forward transform
    'B_2_0'   : 2.06180264956275E-06, # SIP coefficient, forward transform
    'B_2_1'   : 3.45084483143442E-10, # SIP coefficient, forward transform
    'B_3_0'   : -1.3443332487347E-11, # SIP coefficient, forward transform
    'AP_0_2'  : -4.0448921745501E-07, # SIP coefficient, inverse transform
    'AP_0_3'  : 3.21699353088096E-12, # SIP coefficient, inverse transform
    'AP_1_1'  : 6.80045998156693E-06, # SIP coefficient, inverse transform
    'AP_1_2'  : -3.1151183939132E-10, # SIP coefficient, inverse transform
    'AP_2_0'  : 2.08146921677674E-06, # SIP coefficient, inverse transform
    'AP_2_1'  : 1.57988601117428E-11, # SIP coefficient, inverse transform
    'AP_3_0'  : -3.3234678374882E-10, # SIP coefficient, inverse transform
    'BP_0_2'  : 4.77339753614701E-06, # SIP coefficient, inverse transform
    'BP_0_3'  : -3.0367072721136E-10, # SIP coefficient, inverse transform
    'BP_1_1'  : 2.56521902189100E-06, # SIP coefficient, inverse transform
    'BP_1_2'  : 4.16153670376433E-11, # SIP coefficient, inverse transform
    'BP_2_0'  : -2.0621895249828E-06, # SIP coefficient, inverse transform
    'BP_2_1'  : -3.8005816466080E-10, # SIP coefficient, inverse transform
    'BP_3_0'  : -4.0679249990565E-13, # SIP coefficient, inverse transform
}

# From jw01373012001_03103_00003_nrcb3_uncal.fits
header2 = {
    'NAXIS1'  :                 2048,
    'NAXIS2'  :                 2048,
    'CRPIX1'  :               1024.5, # axis 1 coordinate of the reference pixel
    'CRPIX2'  :               1024.5, # axis 2 coordinate of the reference pixel
    'CRVAL1'  :    8.386804280980295, # first axis value at the reference pixel
    'CRVAL2'  :    2.104539390393221, # second axis value at the reference pixel
    'CTYPE1'  : 'RA---TAN'          , # first axis coordinate type
    'CTYPE2'  : 'DEC--TAN'          , # second axis coordinate type
    'CUNIT1'  : 'deg'               , # first axis units
    'CUNIT2'  : 'deg'               , # second axis units
    'CDELT1'  : 8.56364722222222E-06, # first axis increment per pixel
    'CDELT2'  : 8.58809722222222E-06, # second axis increment per pixel
    'PC1_1'   :   0.3170605464082376, # linear transformation matrix element
    'PC1_2'   :  -0.9484052983357431, # linear transformation matrix element
    'PC2_1'   :  -0.9484052983357431, # linear transformation matrix element
    'PC2_2'   :  -0.3170605464082376, # linear transformation matrix element
}


class Test_WCSFOV(unittest.TestCase):

  def runTest(self):

    np.random.seed(9400)

    PolynomialFOV.DEBUG = False
    SpeedTest = False

    for h,header in enumerate((header1, header2)):
        fov = WCSFOV(header, 'y', fast=False)
        fov_fast = WCSFOV(header, 'y', fast=True)

        self.assertTrue(fov.polyfov.max_inversion_error(), 1.e-12)
        self.assertTrue(fov_fast.polyfov.max_inversion_error(), 0.15)

        ############################################
        # Comparison to astropy.wcs.WCS

        uv = Pair.combos(np.arange(4,2048,8),np.arange(4,2048,8))

        wcs = WCS(header)
        uvals = uv.vals[...,0].flatten()
        vvals = uv.vals[...,1].flatten()
        world = wcs.pixel_to_world(uvals - 0.5, vvals - 0.5)
        wcs_ra  = world.ra.value.reshape(uv.shape)      # in degrees
        wcs_dec = world.dec.value.reshape(uv.shape)

        ra_dec = fov.wcs_from_uv(uv)
        fov_ra  = ra_dec.vals[...,0] * DPR
        fov_dec = ra_dec.vals[...,1] * DPR

        self.assertTrue(abs(fov_ra  - wcs_ra ).max() < (3.e-10 if h == 0 else 3.e-5))
        self.assertTrue(abs(fov_dec - wcs_dec).max() < (5.e-08 if h == 0 else 3.e-5))

        los_fov = fov.los_from_uv(uv)
        los_j2000 = fov.cmatrix.transform.matrix.unrotate(los_fov)
        (ra, dec, _) = los_j2000.to_ra_dec_length()

        self.assertTrue(abs(ra.vals  * DPR - wcs_ra ).max() < (4.e-14 if h == 0 else 3.e-5))
        self.assertTrue(abs(dec.vals * DPR - wcs_dec).max() < (4.e-14 if h == 0 else 3.e-5))

        ############################################
        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(4,2048,8),np.arange(4,2048,8))
        uv.insert_deriv('t' , Pair(np.random.randn(uv.size*2)
                                        .reshape(uv.shape + (2,))))
        uv.insert_deriv('rs', Pair(np.random.randn(uv.size*4)
                                        .reshape(uv.shape + (2,2)), drank=1))

        if h == 0 and SpeedTest:
            iters = 4
            t0 = time.time()
            for k in range(iters):
                xy = fov_fast.xy_from_uv(uv, derivs=True)
                uv_test = fov_fast.uv_from_xy(xy, derivs=True)
            t1 = time.time()
            LOGGING.print('fast time = %.2f ms' % ((t1-t0)/iters*1000.))

            t0 = time.time()
            for k in range(iters):
                xy = fov.xy_from_uv(uv, derivs=True)
                uv_test = fov.uv_from_xy(xy, derivs=True)
            t1 = time.time()
            LOGGING.print('slow time = %.2f ms' % ((t1-t0)/iters*1000.))
        else:
            xy = fov.xy_from_uv(uv, derivs=True)
            uv_test = fov.uv_from_xy(xy, derivs=False)

        self.assertTrue(abs(uv - uv_test).max() < 1.e-12)

        EPS = 1.e-6
        xy0 = fov.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0] + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        ############################################
        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(uv.size*2)
                                        .reshape(uv.shape + (2,))))
        xy.insert_deriv('rs', Pair(np.random.randn(uv.size*4)
                                        .reshape(uv.shape + (2,2)), drank=1))
        uv = fov.uv_from_xy(xy, derivs=True)

        xy_test = fov.xy_from_uv(uv)
        self.assertTrue(abs(xy - xy_test).max() < 1.e-14)

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

        DEL = 2.e-6
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        ############################################
        # Same tests, fast=True

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(4,2048,8),np.arange(4,2048,8))
        uv.insert_deriv('t' , Pair(np.random.randn(uv.size*2)
                                        .reshape(uv.shape + (2,))))
        uv.insert_deriv('rs', Pair(np.random.randn(uv.size*4)
                                        .reshape(uv.shape + (2,2)), drank=1))

        xy = fov_fast.xy_from_uv(uv, derivs=True)
        uv_test = fov_fast.uv_from_xy(xy, derivs=False)
        self.assertTrue(abs(uv - uv_test).max() < 0.15)

        EPS = 1.e-6
        xy0 = fov_fast.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov_fast.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov_fast.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov_fast.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0] + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        ############################################
        #### xy -> uv -> xy, with derivs

        xy = fov_fast.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(uv.size*2)
                                        .reshape(uv.shape + (2,))))
        xy.insert_deriv('rs', Pair(np.random.randn(uv.size*4)
                                        .reshape(uv.shape + (2,2)), drank=1))
        uv = fov_fast.uv_from_xy(xy, derivs=True)

        xy_test = fov_fast.xy_from_uv(uv)
        self.assertTrue(abs(xy - xy_test).max() < 0.15)

        EPS = 1.e-6
        xy0 = fov_fast.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov_fast.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov_fast.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov_fast.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0]    + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 2.e-4
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        ############################################
        # Half-resolution test, no derivs

        if h == 0 and SpeedTest:
            uv = Pair.combos(np.arange(1,2048,2),np.arange(1,2048,2))
            iters = 1

            t0 = time.time()
            for k in range(iters):
                xy = fov_fast.xy_from_uv(uv, derivs=False)
                uv_test = fov_fast.uv_from_xy(xy, derivs=False)
            t1 = time.time()
            LOGGING.print('fast time = %.2f ms' % ((t1-t0)/iters*1000.))

            t0 = time.time()
            for k in range(iters):
                xy = fov.xy_from_uv(uv, derivs=False)
                uv_test = fov.uv_from_xy(xy, derivs=False)
            t1 = time.time()
            LOGGING.print('slow time = %.2f ms' % ((t1-t0)/iters*1000.))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
