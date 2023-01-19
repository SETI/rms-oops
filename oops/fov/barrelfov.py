################################################################################
# oops/fov/barrelfov.py: Barrel distortion subclass of FOV
################################################################################

import numpy as np
import sys

from polymath         import Scalar, Pair
from oops.config      import LOGGING
from oops.fov         import FOV
from oops.fov.flatfov import FlatFOV

EPSILON = sys.float_info.epsilon/2.         # actual machine precision

class BarrelFOV(FOV):
    """Subclass of FOV that describes a field of view in which the distortion is
    described by a 1-D polynomial in distance from the image center.
    """

    # True to print convergence steps in _solve_polynomial()
    DEBUG = False

    #===========================================================================
    def __init__(self, uv_scale, uv_shape, coefft_xy_from_uv=None,
                 coefft_uv_from_xy=None, uv_los=None, uv_area=None,
                 iters=8, fast=True):
        """Constructor for a BarrelFOV.

        Inputs:
            uv_scale    a single value, tuple or Pair defining the ratios dx/du
                        and dy/dv. At the center of the FOV.  For example, if
                        (u,v) are in units of  arcseconds, then
                            uv_scale = Pair((pi/180/3600.,pi/180/3600.))
                        Use the sign of the second element to define the
                        direction of increasing V: negative for up, positive for
                        down.

            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            coefft_xy_from_uv
                        the polynomial coefficient array describing the radial
                        distortion from U,V to X,Y. It is a function of r,
                        defined as
                            r = sqrt(((u-uv_los[0]) * uv_scale[0])**2 +
                                     ((v-uv_los[1]) * uv_scale[1])**2))
                        In other words, r is in units of radians and measures
                        the distance from the center of the FOV if there were no
                        distortion. The polynomial f(r) returns the distorted
                        distance given the un-distorted distance. Because this
                        polynomial cannot have a constant term, the coefficients
                        begin with the linear term, which is typically ~ 1. In
                        other words, coefft_xy_from_uv[i] is the coefficient on
                        r**(i+1). If this input is None, the distortion
                        polynomial for uv_from_xy is inverted.

            coefft_uv_from_xy
                        the polynomial coefficient array describing the radial
                        distortion scale factor from X,Y to U,V. It is a
                        function of r, defined as
                            r = sqrt(x**2 + y**2),
                        in units of radians. The array has shape (order,) under
                        the assumption that there can be no constant term, so
                        coefft_uv_from_xy[i] is the coefficient on r**(i+1). The
                        first coefficient is typically ~ 1, implying no
                        distortion at the center of the FOV. If None, the
                        distortion polynomial for xy_from_uv is inverted.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal area of a
                        pixel in steradians after distortion has been removed.

            iters       the number of iterations of Newton's method to use when
                        inverting the distortion polynomial.

            fast        if True and both sets of coefficients are provided, the
                        polynomials will be used in both directions, meaning
                        that the conversions xy_from_uv and uv_from_xy might be
                        inconsistent, although probably at the sub-pixel level.
                        If False, then uv_from_xy is refined further using one
                        or two steps of Newton's method, which provides
                        consistency at the level of machine precision, but
                        uv_from_xy will be somewhat slower.
        """

        self.coefft_xy_from_uv = None
        self.coefft_uv_from_xy = None

        # Save the coefficients
        #
        # The function we evaluate is actually polynomial(r)/r, which is very well
        # behaved (nearly constant) in both directions.

        if coefft_xy_from_uv is not None:
            order = len(coefft_xy_from_uv)
            self.coefft_xy_from_uv = np.asfarray(coefft_xy_from_uv)
            self.dcoefft_xy_from_uv = (self.coefft_xy_from_uv *
                                       np.arange(order))

        if coefft_uv_from_xy is not None:
            order = len(coefft_uv_from_xy)
            self.coefft_uv_from_xy = np.asfarray(coefft_uv_from_xy)
            self.dcoefft_uv_from_xy = (self.coefft_uv_from_xy *
                                       np.arange(order))

        if (self.coefft_xy_from_uv is None and
            self.coefft_uv_from_xy is None):
                raise ValueError('at least one of coefft_xy_from_uv and '
                                 + 'coefft_uv_from_xy must be specified')

        self.uv_scale = Pair.as_pair(uv_scale).as_readonly()
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float()
            self.uv_los.as_readonly()

        self.iters = max(int(iters), 2)
        self.fast = bool(fast) and (self.coefft_uv_from_xy is not None)

        self.flat_fov = FlatFOV(self.uv_scale, self.uv_shape, self.uv_los)

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        # Reference values for precision determinations
        # The goal is full precision in pixel coordinates
        self.uv_precision = EPSILON
        self.xy_precision = EPSILON * np.min(self.uv_scale.vals)

    def __getstate__(self):
        return (self.uv_scale, self.uv_shape, self.coefft_xy_from_uv,
                self.coefft_uv_from_xy, self.uv_los, self.uv_area, self.iters)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv, time=None, derivs=False, remask=False):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv          (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute times. Ignored by BarrelFOV.
            derivs      if True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        # Convert to xy using flat FOV model
        flat_xy = self.flat_fov.xy_from_uv(uv, derivs=derivs, remask=remask)
        r_flat = flat_xy.norm(derivs)

        # Distort based on which types of coefficients are given
        if self.coefft_xy_from_uv is not None:
            true_over_flat = BarrelFOV._eval_ratio(r_flat,
                                                   self.coefft_xy_from_uv,
                                                   self.dcoefft_xy_from_uv,
                                                   derivs=derivs)
        else:
            r_true_guess = r_flat.wod
            true_over_flat = BarrelFOV._solve_ratio(r_flat, r_true_guess,
                                                    self.coefft_uv_from_xy,
                                                    self.dcoefft_uv_from_xy,
                                                    derivs=derivs,
                                                    iters=self.iters,
                                                    precision=self.xy_precision)

        return flat_xy * true_over_flat

    #===========================================================================
    def uv_from_xyt(self, xy, time=None, derivs=False, remask=False):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy          (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute times. Ignored by BarrelFOV.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        true_xy = Pair.as_pair(xy, derivs)
        r_true = true_xy.norm(derivs)

        # Distort based on which types of coefficients are given
        if self.fast and self.coefft_uv_from_xy is not None:
            flat_over_true = BarrelFOV._eval_ratio(r_true,
                                                   self.coefft_uv_from_xy,
                                                   self.dcoefft_uv_from_xy,
                                                   derivs=derivs)
        else:
            # If both sets of coefficients are available, use uv_from_xy as the
            # guess. Otherwise, use a flat FOV
            if self.coefft_uv_from_xy is not None:
                flat_over_true = BarrelFOV._eval_ratio(r_true,
                                                       self.coefft_uv_from_xy,
                                                       self.dcoefft_uv_from_xy,
                                                       derivs=False)
                r_flat_guess = r_true.wod / flat_over_true
            else:
                r_flat_guess = r_true.wod

            flat_over_true = BarrelFOV._solve_ratio(r_true, r_flat_guess,
                                                    self.coefft_xy_from_uv,
                                                    self.dcoefft_xy_from_uv,
                                                    derivs=derivs,
                                                    iters=self.iters,
                                                    precision=self.uv_precision)

        flat_xy = true_xy * flat_over_true
        return self.flat_fov.uv_from_xy(flat_xy, derivs=derivs, remask=remask)

    #===========================================================================
    @staticmethod
    def _eval_ratio(r, coefft, dcoefft, derivs=False, d_dr=False):
        """Compute the ratio polynomial(r)/r.

        By returning the ratio instead of the polynomial value directly, it is
        easier to handle r = polynomial(r) = 0.

        Input:
            r           Scalar of arbitrary shape specifying the points at which
                        to evaluate the polynomial.
            coefft      The coefficient array defining the polynomial, with the
                        leading zero-valued constant term omitted.
            dcoefft     The coefficients of the derivatives of the ratio, i.e.,
                            coefft * [0,1,2,...]
            derivs      True to include the derivatives embedded in r in the
                        result.
            d_dr        If True, the returned quantity is a tuple (f, df/dr);
                        otherwise, only f is returned.

        Return          ratio or (ratio, dratio_dr), depending on d_dr input.
            ratio       value of the polynomial(r)/r.
            dratio_dr   optional derivative of the ratio with respect to r.
        """

        # Construct the powers of radius, starting at 1
        r = Scalar.as_scalar(r, derivs)

        powers = np.empty(r.shape + coefft.shape)
        powers[...,0] = 1.
        powers[...,1] = r.vals
        for k in range(2, coefft.shape[0]):
            powers[...,k] = powers[...,k-1] * r.vals

        # Evaluate the polynomial; start from higher order to (maybe) improve
        ratio = Scalar(np.sum(powers * coefft, axis=-1), r.mask)

        # Evaluate the derivative with respect to r if necessary
        # Note that dcoefft[0] is always 0.
        if d_dr or derivs:
            dratio_dr = Scalar(np.sum(dcoefft[1:] * powers[...,:-1],
                               axis=-1))    # unmasked is OK

        # Calculate additional derivatives if necessary
        if derivs:
            new_derivs = {}
            for key, deriv in r.derivs.items():
                new_derivs[key] = dratio_dr * deriv
            ratio.insert_derivs(new_derivs)

        if d_dr:
            return (ratio, dratio_dr)
        else:
            return ratio

    #===========================================================================
    @staticmethod
    def _solve_ratio(f, r_guess, coefft, dcoefft, derivs=False, iters=8,
                                                                precision=0.):
        """Invert a 1-D polynomial to find r where polynomial(r) = f, but then
        return r/f.

        Using the ratio r/f instead of r itself makes it easier to handle
        r = f = 0.

        Input:
            f           Scalar of arbitrary shape specifying the values of the
                        polynomial.
            r_guess     initial guess at the values to return.
            coefft      coefficient array defining the polynomial, with the
                        leading zero-valued constant term omitted.
            dcoefft     coefficients of the derivatives, i.e.,
                            coefft[1:] * [1,2,3,...]
            derivs      True to include the derivatives embedded in f in the
                        result.
            d_dr        if True, the tuple (r, d_dr) is returned insteadu of r
                        alone.
            iters       maximum number of iterations of Newton's method.
            precision   absolute precision desired. Approximate limit is OK, and
                        the only down-side of zero (the default) is that the
                        solution will require one extra iteration.

        Output:         r or (r, df_dr) depending on input d_dr.
            r           Scalar of the same shape as f giving the values at which
                        the polynomial evaluates to f.
            df_dr       the derivative df/dr at r.
        """

        f = Scalar.as_scalar(f, derivs)

        # Handle fully-masked case
        if np.all(f.mask):
            return Pair(np.ones(f.shape), True)

        # Because convergence is quadratic in Newton's method, once we get half-
        # way to convergence, the next iteration should be exact.
        eps = 2*[precision * 2] + (iters-2) * [np.sqrt(precision) / 30]
            # Don't assume the convergence is quadratic till the third iteration
            # Division by 30 is just for extra safety

        # Make sure the initial r guess is an array copy and uses f's mask
        r = Scalar(r_guess.vals.copy(), f.mask)

        max_dr = 1.e99
        converged = False
        for count in range(iters):
            (f_over_r,
             d_f_over_r_dr) = BarrelFOV._eval_ratio(r, coefft, dcoefft,
                                                       derivs=False, d_dr=True)
            f_test = f_over_r * r
            df_dr = f_over_r + r * d_f_over_r_dr

            # Perform one step of Newton's Method
            dr = (f.wod - f_test) / df_dr
                # Note that df_dr should never be zero, so this is safe
            new_max_dr = abs(dr).max(builtins=True, masked=-1.)

            if LOGGING.fov_iterations or BarrelFOV.DEBUG:
                LOGGING.convergence('BarrelFOV._solve_ratio:',
                                    'iter=%d; change=%.6g' % (count+1,
                                                              new_max_dr))

            # Quit when convergence stops
            if new_max_dr <= eps[count]:
                r += dr
                converged = True
                break

            if new_max_dr >= max_dr:
                break

            r += dr
            max_dr = new_max_dr

        if not converged:
            LOGGING.warn('BarrelFOV._solve_ratio did not converge;',
                         'iter=%d; change=%.6g' % (count+1, max_dt))

        # Prepare ratio r/f
        ratio = 1. / f_over_r       # f_over_r can't be zero

        # Propagate derivatives if necessary
        if derivs:
            new_derivs = {}
            for key, df_dx in f.derivs.items():

                # We need to obtain dratio_dx while avoiding divide-by-zero

                dr_dx = df_dx / df_dr           # df_dr cannot equal zero

                # d(ratio)/dx = d(1/f_over_r)/dx
                #   = -d(f_over_r)/dx / f_over_r**2
                #   = -d(f_over_r)/dr * dr/dx / f_over_r**2

                new_derivs[key] = -d_f_over_r_dr * dr_dx * ratio**2

            ratio.insert_derivs(new_derivs)

        return ratio

################################################################################
# UNIT TESTS
################################################################################

import unittest
import time

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
