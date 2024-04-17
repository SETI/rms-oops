################################################################################
# oops/fov/wcsfov.py: FOV subclass for WCS/SIP FOVs described by FITS headers.
################################################################################

import numpy as np

from polymath               import Pair, Matrix
from oops.fov               import FOV
from oops.fov.flatfov       import FlatFOV
from oops.fov.polynomialfov import PolynomialFOV
from oops.frame.cmatrix     import Cmatrix
from oops.constants         import RPD, DPR

class WCSFOV(FOV):
    """PolynomialFOV subclass represented by WCS SIP parameters in a FITS
    header.

    The FITS WCS parameters define both the camera distortion and also image's
    instantaneous (aberration-corrected) pointing. We need to decouple these two
    for our purposes.

    In addition to the standard FOV attributes (uv_los, uv_scale, uv_shape, and
    uv_area), a WCSFOV also has these:

        header      given FITS header.
        ref_axis    reference axis, "x" or "y". This is the image axis used to
                    define the FOV's frame. It is needed because the axes can
                    be slightly skewed.
        ra          apparent right ascension at reference point in image, in
                    radians.
        dec         apparent declination at references point in image, in
                    radians.
        clock       clock angle for celestial north in the image, in radians.
        ref_uv      reference point in pixel coordinates, as a Pair object.
        cmatrix     Cmatrix frame that rotates apparent J2000 coordinates into
                    the camera FOV, based on the WCS parameters provided. It is
                    useful for consistency testing and as a starting point for
                    a Frame definition.
    """

    #===========================================================================
    def __init__(self, header, ref_axis='y', fast=True):
        """Constructor for a, FOV object that handles the WCS parameters from a
        FITS header.

        Inputs:
            header      FITS header (or equivalent dictionary).
            ref_axis    "x" or "y", the axis to align with the FOV frame if the
                        axes are not exactly perpendicular.
            fast        if True and the WCS model includes the coefficients for
                        the reverse transform (xy to uv), then that reverse
                        transform will be used as is. If False, it will instead
                        be used as the starting point for an exact reverse
                        transform using Newton's method.

        """

        # Description of the WCS parameters is found here:
        # David L. Shupe and Richard N. Hook 2005. The SIP Convention for
        # Representing Distortion in FITS Image Headers. ASP Conference Series,
        # Vol. XXX, P3.2.18, 5 pp.

        self.header = header
        self.ref_axis = str(ref_axis)
        self.fast = bool(fast)

        if ref_axis not in ('x', 'y'):
            raise ValueError('invalid value of ref_axis: ' + repr(ref_axis))

        self.uv_shape = Pair([header['NAXIS2'], header['NAXIS1']])
        self.uv_los = Pair([header['CRPIX1'] - 0.5, header['CRPIX2'] - 0.5])

        # We require that x_wcs = RA and y_wcs = dec
        if (header['CTYPE1'][:8] != 'RA---TAN' or
            header['CTYPE2'][:8] != 'DEC--TAN'):
                raise ValueError('only WCS CTYPEs "RA---TAN" and "DEC--TAN" '
                                 + 'are supported')
        if header['CUNIT1'] != 'deg' and  header['CUNIT2'] != 'deg':
            raise ValueError('only CUNIT = "deg" is supported')

        # The FITS formula is:
        #   (x_wcs, y_wcs) = [[CD1_1,CD1_2],[CD2_1,CD2_2]] (u+f(u,v), v+g(u,v))
        # where
        #   f(u,v) = Sum(A_p_q u**p v**q) for p + q <= A_ORDER
        #   g(u,v) = Sum(B_p_q u**p v**q) for p + q <= B_ORDER

        # Get the coefficients
        if 'A_ORDER' in header:
            ab = WCSFOV._sips_coefficients(header, ['A', 'B'])
            if 'AP_ORDER' in header:
                abp = WCSFOV._sips_coefficients(header, ['AP', 'BP'])
            else:
                abp = None

            self.polyfov = PolynomialFOV(self.uv_shape, ab, abp,
                                         uv_los=self.uv_los, fast=self.fast)
        else:       # without a distortion model, we use a FlatFOV instead
                    # (in spite of the attribute name)
            self.polyfov = FlatFOV(1., self.uv_shape, uv_los=self.uv_los)

        # Now we need to handle the CD matrix, which contains scale, rotation,
        # and skew. We need to decouple the rotation component from the others,
        # because the rotation is handled will be handled by the instrument
        # frame; it is not an aspect of the FOV.
        #
        # Let's define u' = u + f(u,v); v' = v + g(u,v), so we have:
        #   (x_wcs, y_wcs) = CD (u', v')
        #
        # We will also need to support the inverse transform, in which:
        #   (U, V) = CD-inverse (x_wcs, y_wcs)
        #   u = U + F(U,V)
        #   v = V + G(U,V)
        # Here, F and G have the same polynomial representations as functions
        # f and g, but with a different set of coefficients.
        #
        # Our solution is to replace CD by an "un-rotated" matrix CD', in which
        # the (u', v') axes are aligned with the (x_wcs, y_wcs) axes:
        #   CD' = R * CD
        # where R is a right-handed rotation matrix:
        #   R = [[cos(c), sin(c)], [-sin(c), cos(c)]]
        # Here, c is the clock angle, i.e., the angle of celestial north in the
        # image as measured clockwise from the v' axis.
        #
        # Let's write this all out for convenience:
        #   CD1_1' =  cos(c) * CD1_1 + sin(c) * CD2_1   (1a)
        #   CD1_2' =  cos(c) * CD1_2 + sin(c) * CD2_2   (1b)
        #   CD2_1' = -sin(c) * CD1_1 + cos(c) * CD2_1   (1c)
        #   CD2_2' = -sin(c) * CD1_2 + cos(c) * CD2_2   (1d)
        #
        # OPTION 1: Force y_wcs and v' to be parallel. This requires:
        #   CD2_1' = 0
        #   CD2_2' > 0
        # From (1c):
        #   -sin(c) * CD1_1 + cos(c) * CD2_1 = 0
        #   sin(c) = CD2_1/CD1_1 * cos(c)
        #   tan(c) = CD2_1/CD1_1
        # From (1d):
        #   -sin(c) * CD1_2 + cos(c) * CD2_2 > 0
        #   -CD2_1/CD1_1 * cos(c) * CD1_2 + cos(c) * CD2_2 > 0
        #   cos(c) * (CD2_2 - CD1_2*CD2_1/CD1_1) > 0
        # Therefore, cos(c) and (CD2_2 - CD1_2*CD2_1/CD1_1) have the same sign.
        #
        # We know that CD has a negative determinant, because it converts
        # right-handed (u',v') to left-handed (x_wcs, y_wcs). Therefore,
        #   CD1_1 * CD2_2 - CD1_2 * CD2_1 < 0
        # It follows that CD1_1 and (CD2_2 - CD1_2*CD2_1/CD1_1) have opposite
        # signs, so cos(c) and CD1_1 have opposite signs.
        #
        # This definition ensures that tan(c) = CD2_1/CD1_1 but that CD1_1 and
        # cos(c) have opposite signs:
        #   c = np.arctan2(-CD2_1, -CD1_1)
        #
        # OPTION 2: Force x_wcs and u' to be anti-parallel. This requires:
        #   CD1_1' < 0
        #   CD1_2' = 0
        # From (1b):
        #   cos(c) * CD1_2 + sin(c) * CD2_2 = 0
        #   sin(c) = -CD1_2/CD2_2 * cos(c)
        #   tan(c) = -CD1_2/CD2_2
        # From (1a):
        #   cos(c) * CD1_1 + sin(c) * CD2_1 < 0
        #   cos(c) * CD1_1 - CD1_2/CD2_2 * cos(c) * CD2_1 < 0
        #   cos(c) * (CD1_1 - CD1_2*CD2_1/CD2_2) < 0
        # Therefore, cos(c) and (CD1_1 - CD1_2*CD2_1/CD2_2) have opposite signs.
        #
        # Using similar reasoning as above, cos(c) and CD2_2 must have the same
        # sign, so this follows:
        #   c = np.arctan2(-CD1_2, CD2_2)

        if 'CD1_1' in header:
            cd11 = header['CD1_1'] * RPD
            cd12 = header['CD1_2'] * RPD
            cd21 = header['CD2_1'] * RPD
            cd22 = header['CD2_2'] * RPD
        else:
            # From http://montage.ipac.caltech.edu/docs/headers.html
            #
            # x = (i-CRPIX1)*CD1_1 + (j-CRPIX2)*CD1_2
            # y = (i-CRPIX1)*CD2_1 + (j-CRPIX2)*CD2_2
            #
            # x = CDELT1*(i-CRPIX1)*PC1_1 + CDELT2*(j-CRPIX2)*PC1_2
            # y = CDELT1*(i-CRPIX1)*PC2_1 + CDELT2*(j-CRPIX2)*PC2_2

            cdelt1 = header['CDELT1'] * RPD
            cdelt2 = header['CDELT2'] * RPD
            cd11 = header['PC1_1'] * cdelt1
            cd12 = header['PC1_2'] * cdelt2
            cd21 = header['PC2_1'] * cdelt1
            cd22 = header['PC2_2'] * cdelt2

        self.cd = Matrix([[cd11, cd12], [cd21, cd22]])

        if ref_axis == 'x':
            self.clock = np.arctan2(-cd12, cd22)
        else:
            self.clock = np.arctan2(-cd21, -cd11)

        # From (1a-d) above...
        #   CD1_1' =  cos(c) * CD1_1 + sin(c) * CD2_1   (1a)
        #   CD1_2' =  cos(c) * CD1_2 + sin(c) * CD2_2   (1b)
        #   CD2_1' = -sin(c) * CD1_1 + cos(c) * CD2_1   (1c)
        #   CD2_2' = -sin(c) * CD1_2 + cos(c) * CD2_2   (1d)

        cos_clock = np.cos(self.clock)
        sin_clock = np.sin(self.clock)
        self.rotmat = Matrix([[cos_clock, sin_clock], [-sin_clock, cos_clock]])

        self.cdp = self.rotmat * self.cd
        self.cdp_inv = self.cdp.inverse()

        # At this point, CD' and CD'-inverse can take the place of CD and
        # CD-inverse in the WCS formulas.

        # Prepare to fix the sign discrepancy: x_wcs = -x_fov; y_wcs = -y_fov
        self.neg_cdp = -self.cdp
        self.neg_cdp_inv = -self.cdp_inv

        # Define the remaining required attributes
        self.uv_scale = Pair(np.diag(self.neg_cdp.vals))
        self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])

        # Create the nominal rotation frame J2000 to the FOV
        self.ra  = header['CRVAL1'] * RPD
        self.dec = header['CRVAL2'] * RPD

        self.cmatrix = Cmatrix.from_ra_dec(self.ra * DPR,
                                           self.dec * DPR,
                                           -self.clock * DPR)

    #===========================================================================
    def __getstate__(self):
        return (self.header, self.ref_axis, self.fast)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def _sips_coefficients(header, prefixes):
        """Internal method to read SIPS coefficients from the header."""

        # Determine polynomial order
        order = 1
        for prefix in prefixes:
            order = max(order, header[prefix + '_ORDER'])

        # Create an empty array for the coefficients
        coeffts = np.zeros((order+1, order+1, len(prefixes)))
        coeffts[1,0,0] = 1.
        coeffts[0,1,1] = 1.

        # The first index is the order of the term.
        # The second index is the coefficient on the u-axis.
        for k, prefix in enumerate(prefixes):
          for i in range(order+1):
            for j in range(order+1 - i):
                try:
                    coeffts[i,j,k] = header['%s_%d_%d' % (prefix, i, j)]
                except KeyError:
                    pass

        return coeffts

    #===========================================================================
    def xy_from_uvt(self, uv, time=None, derivs=False, remask=False):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv          (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute times. Ignored by WCSFOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        return self.neg_cdp * self.polyfov.xy_from_uvt(uv, derivs=derivs,
                                                           remask=remask)

    #===========================================================================
    def uv_from_xyt(self, xy, time=None, derivs=False, remask=False):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy          (x,y) Pair in FOV coordinates.
            tfrac       Scalar of fractional times during the exposure. Ignored
                        by WCSFOV.
            time        Scalar of optional absolute times. Ignored by
                        WCSFOV.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        xy0 = self.neg_cdp_inv * Pair.as_pair(xy, recursive=derivs)
        return self.polyfov.uv_from_xy(xy0, derivs=derivs, remask=remask)

    #===========================================================================
    def wcs_from_uv(self, uv, derivs=False, remask=False):
        """The WCS coordinates (apparent RA, dec) given the FOV coordinates
        (u,v).

        This bypasses any coordinate frame to return the (RA,dec) offsets as
        indicated in the FITS header. It should mimic
            astropy.wcs.WCS.pixel_to_world(u - 0.5, v - 0.5) * RPD

        Input:
            uv          (u,v) coordinate Pair in the FOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        radec = self.cd * self.polyfov.xy_from_uv(uv, derivs=derivs,
                                                      remask=remask)

        # Add the offset and fix up the RA based on the declination
        radec.vals[...,1] += self.dec
        cos_dec = np.cos(radec.vals[...,1])

        radec.vals[...,0] /= cos_dec
        for key, deriv in radec.derivs:
            deriv.vals[...,0] /= cos_dec

        radec.vals[...,0] += self.ra

        return radec

################################################################################
# UNIT TESTS
################################################################################

import unittest

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

    import time

    # Importing astropy.wcs.WCS raises a RuntimeWarning unless import is at the
    # top level, but I don't want to import it except during unit tests. This
    # solves the problem.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from astropy.wcs import WCS

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
