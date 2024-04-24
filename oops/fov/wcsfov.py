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
        if header['CUNIT1'] != 'deg' and header['CUNIT2'] != 'deg':
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
