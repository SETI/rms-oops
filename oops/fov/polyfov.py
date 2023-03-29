################################################################################
# oops/fov/polyfov.py: PolyFOV subclass of FOV, and WCS FOV support.
################################################################################

import numpy as np

from polymath         import Pair
from oops.config      import LOGGING
from oops.fov         import FOV
from oops.fov.flatfov import FlatFOV

import sys
EPSILON = sys.float_info.epsilon/2.         # actual machine precision

class PolyFOV(FOV):
    """Subclass of FOV that describes a field of view in which the distortion is
    described by a 2-D polynomial.

    This is the approached used by Space Telescope Science Institute to describe
    the Hubble instrument fields of view. A PolyFOV has no dependence on
    the optional extra indices that can be associated with time, wavelength
    band, etc.
    """

    DEBUG = False       # Set True to print convergence steps of Newton's Method

    #===========================================================================
    def __init__(self, uv_shape, coefft_xy_from_uv=None,
                 coefft_uv_from_xy=None, uv_los=None, uv_area=None,
                 iters=8, fast=True):
        """Constructor for a PolyFOV.

        Inputs:
            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            coefft_xy_from_uv
                        the coefficient array of the polynomial to convert U,V
                        to X,Y. The array has shape (order+1,order+1,2), where
                        coefft[i,j,0] is the coefficient on (u**i * v**j)
                        yielding x(u,v), and coefft[i,j,1] is the coefficient
                        yielding y(u,v). If None, then the polynomial for
                        uv_from_xy is inverted.

            coefft_uv_from_xy
                        the coefficient array of the polynomial to convert X,Y
                        to U,V. The array has shape (order+1,order+1,2), where
                        coefft[i,j,0] is the coefficient on (x**i * y**j)
                        yielding u(x,y), and coefft[i,j,1] is the coefficient
                        yielding v(x,y). If None, then the polynomial for
                        xy_from_uv is inverted.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal area of a
                        pixel in steradians after distortion has been removed.

            iters       the number of iterations of Newton's method to use when
                        inverting the polynomial; default is 8.

            fast        if True and both sets of coefficients are provided, the
                        polynomials will be used in both directions, meaning
                        that the conversions xy_from_uv and uv_from_xy might be
                        inconsistent, although probably at the sub-pixel level.
                        If False, then uv_from_xy is refined further using one
                        or two steps of Newton's method; which provides
                        consistency at the level of machine precision, but
                        uv_from_xy will be slightly slower.
        """

        # Prepare coefficients
        self.coefft_xy_from_uv = None
        self.coefft_uv_from_xy = None

        if coefft_xy_from_uv is not None:
            self.coefft_xy_from_uv = np.asfarray(coefft_xy_from_uv)
            order = self.coefft_xy_from_uv.shape[0] - 1
            self.coefft_dxy_du = (self.coefft_xy_from_uv[1:] *
                                  np.arange(1,order+1)[:,np.newaxis,np.newaxis])
            self.coefft_dxy_dv = (self.coefft_xy_from_uv[:,1:] *
                                  np.arange(1,order+1)[np.newaxis,:,np.newaxis])

        if coefft_uv_from_xy is not None:
            self.coefft_uv_from_xy = np.asfarray(coefft_uv_from_xy)
            order = self.coefft_uv_from_xy.shape[0] - 1
            self.coefft_duv_dx = (self.coefft_uv_from_xy[1:] *
                                  np.arange(1,order+1)[:,np.newaxis,np.newaxis])
            self.coefft_duv_dy = (self.coefft_uv_from_xy[:,1:] *
                                  np.arange(1,order+1)[np.newaxis,:,np.newaxis])

        if (self.coefft_xy_from_uv is None and
            self.coefft_uv_from_xy is None):
                raise ValueError('at least one of coefft_xy_from_uv and '
                                 + 'coefft_uv_from_xy must be specified')

        self.iters = max(int(iters), 2)
        self.fast = bool(fast) and (self.coefft_uv_from_xy is not None)

        # Required attributes uv_shape and uv_los
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float()
            self.uv_los.as_readonly()

        # Required attribute uv_scale...

        # This is a first guess at flat_fov
        if self.coefft_uv_from_xy is None:
            uv_scale = Pair.as_pair((self.coefft_xy_from_uv[1,0,0],
                                     self.coefft_xy_from_uv[0,1,1]))
        else:
            uv_scale = Pair.as_pair((1./self.coefft_uv_from_xy[1,0,0],
                                     1./self.coefft_uv_from_xy[0,1,1]))

        self.flat_fov = FlatFOV(uv_scale, self.uv_shape, self.uv_los)
        self.uv_precision = EPSILON
        self.xy_precision = EPSILON * min(abs(uv_scale.vals))

        # This is a refined estimate of flat_fov
        (u0, v0) = 0.2 * self.uv_shape.vals
        (u1, v1) = 0.5 * self.uv_shape.vals
        (u2, v2) = 0.8 * self.uv_shape.vals

        p0 = Pair((u0, v1))
        p2 = Pair((u2, v1))
        x0 = self.xy_from_uvt(p0).vals[0]
        x2 = self.xy_from_uvt(p2).vals[0]
        dx_du = (x2 - x0) / (u2 - u0)

        p0 = Pair((u1, v0))
        p2 = Pair((u1, v2))
        y0 = self.xy_from_uvt(p0).vals[1]
        y2 = self.xy_from_uvt(p2).vals[1]
        dy_dv = (y2 - y0) / (v2 - v0)

        self.uv_scale = Pair((dx_du, dy_dv))
        self.flat_fov = FlatFOV(self.uv_scale, self.uv_shape, self.uv_los)

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        # Reference values for precision determinations
        # The goal is full precision in pixel coordinates
        self.uv_precision = EPSILON
        self.xy_precision = EPSILON * min(dx_du, abs(dy_dv))

    def __getstate__(self):
        return (self.uv_shape, self.coefft_xy_from_uv, self.coefft_uv_from_xy,
                self.uv_los, self.uv_area, self.iters)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv, time=None, derivs=False, remask=False):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv          (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute times. Ignored by PolyFOV.
            derivs      if True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        # Mask if necessary
        uv = Pair.as_pair(uv, recursive=derivs)
        if remask:
            uv = uv.mask_or(self.uv_is_outside(uv).vals)

        # Subtract off the center of the field of view
        duv = uv - self.uv_los

        # Transform based on which types of coefficients are given
        if self.coefft_xy_from_uv is not None:
            xy = PolyFOV._eval_polynomial(duv,
                                          self.coefft_xy_from_uv,
                                          self.coefft_dxy_du,
                                          self.coefft_dxy_dv,
                                          derivs=derivs)
        else:
            xy_guess = self.flat_fov.xy_from_uv(uv, derivs=False)
            xy = PolyFOV._solve_polynomial(duv, xy_guess,
                                           self.coefft_uv_from_xy,
                                           self.coefft_duv_dx,
                                           self.coefft_duv_dy,
                                           derivs=derivs,
                                           iters=self.iters,
                                           precision=self.xy_precision)

        return xy

    #===========================================================================
    def uv_from_xyt(self, xy, time=None, derivs=False, remask=False):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy          (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute times. Ignored by PolyFOV.
            derivs      if True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        xy = Pair.as_pair(xy, recursive=derivs)

        # Transform based on which types of coeffs are given
        if self.fast and self.coefft_uv_from_xy is not None:
            duv = PolyFOV._eval_polynomial(xy,
                                           self.coefft_uv_from_xy,
                                           self.coefft_duv_dx,
                                           self.coefft_duv_dy,
                                           derivs=derivs)

        else:
            # If both sets of coefficients are available, use uv_from_xy as the
            # guess. Otherwise, use a flat FOV
            if self.coefft_uv_from_xy is not None:
                duv_guess = PolyFOV._eval_polynomial(xy,
                                                     self.coefft_uv_from_xy,
                                                     self.coefft_duv_dx,
                                                     self.coefft_duv_dy,
                                                     derivs=False)
            else:
                duv_guess = (self.flat_fov.uv_from_xy(xy, derivs=False)
                             - self.uv_los)

            # Use the xy_from_uv coefficients to ensure that the polynomial
            # inversion is exact.
            duv = PolyFOV._solve_polynomial(xy, duv_guess,
                                            self.coefft_xy_from_uv,
                                            self.coefft_dxy_du,
                                            self.coefft_dxy_dv,
                                            derivs=derivs,
                                            iters=self.iters,
                                            precision=self.uv_precision)

        # Add back the center of the field of view
        uv = duv + self.uv_los

        # Mask if necessary
        if remask:
            uv = uv.mask_or(self.uv_is_outside(uv).vals)

        return uv

    #===========================================================================
    @staticmethod
    def _eval_polynomial(pq, coefft, dcoefft_p, dcoefft_q,
                         derivs=False, d_dpq=False):
        """Evaluate the polynomial at pair (p,q) to return (a,b).

        Input:
            pq          Pairs of arbitrary shape specifying the points at which
                        to evaluate the polynomial.
            coefft      coefficient array defining the polynomial.
            dcoefft_p   coefficient array for the polynomial derivative with
                        respect to p.
            dcoefft_q   coefficient array for the polynomial derivative with
                        respect to q.
            derivs      if True, derivatives are computed and included in the
                        result.
            d_dpq       if True, the returned quantity is a tuple (f, df/dpq);
                        otherwise, only f is returned.

        Return          ab or (ab, dab_dpq), depending on df_dpq input.
            ab          value of the polynomial.
            dab_dpq     optional derivative of f with respect to pq.
        """

        pq = Pair.as_pair(pq, derivs)

        # Start with empty buffer
        order_plus_1 = coefft.shape[0]
        powers = np.empty((order_plus_1, order_plus_1) + pq.shape)

        p = pq.vals[...,0]
        q = pq.vals[...,1]

        # Fill in powers[:,0] with powers of p
        powers[0,0] = 1.
        powers[1,0] = p
        for k in range(2, order_plus_1):
            powers[k,0] = powers[k-1,0] * p

        # Fill in powers[:,1] with q times powers of p
        powers[0,1] = q     # skip an unnecessary multiply by one
        powers[1:,1] = q * powers[1:,0]

        # Fill in powers[:,2:] with q times powers[:,1:]
        for k in range(2, order_plus_1):
            powers[:,k] = q * powers[:,k-1]

        # Rotate the leading axes to the end
        powers = np.moveaxis(powers, (0,1), (-2,-1))[..., np.newaxis]

        # Evaluate the polynomials
        ab = Pair(np.sum(coefft * powers, axis=(-3,-2)), pq.mask)

        # Evaluate the derivatives with respect to pq if necessary
        if d_dpq or derivs:
            dab_dpq_vals = np.empty((2,) + pq.vals.shape)
            _ = np.sum(dcoefft_p * powers[...,:-1,:,:], axis=(-3,-2),
                       out=dab_dpq_vals[0])
            _ = np.sum(dcoefft_q * powers[...,:,:-1,:], axis=(-3,-2),
                       out=dab_dpq_vals[1])
            dab_dpq = Pair(np.moveaxis(dab_dpq_vals, 0, -1), drank=1)

        # Calculate additional derivatives if necessary
        if derivs:
            new_derivs = {}
            for key, deriv in pq.derivs.items():
                new_derivs[key] = dab_dpq.chain(deriv)
            ab.insert_derivs(new_derivs)

        if d_dpq:
            return (ab, dab_dpq)
        else:
            return ab

    #===========================================================================
    @staticmethod
    def _solve_polynomial(ab, pq_guess, coefft, dcoefft_p, dcoefft_q,
                          derivs=False, iters=8, precision=0.):
        """Solve the polynomial for an (a,b) pair to return (p,q).

        Input:
            ab          Pair of arbitrary shape specifying the values of the
                        polynomial.
            pq_guess    initial guess at the values to return.
            coefft      coefficient array defining the polynomial.
            dcoefft_p   coefficient array for the polynomial derivative with
                        respect to p.
            dcoefft_q   coefficient array for the polynomial derivative with
                        respect to q.
            derivs      if True, derivatives are included in the output.
            iters       maximum number of iterations of Newton's method.
            precision   absolute precision desired. Approximate limit is OK, and
                        the only down-side of zero (the default) is that the
                        solution will require one extra iteration.

        Output:         Pair of the same shape as ab giving the values at which
                        the polynomial evaluates to ab.
        """

        ab = Pair.as_pair(ab, derivs)

        # Handle fully-masked case
        if np.all(ab.mask):
            return Pair(np.zeros(ab.shape), True)

        # Because convergence is quadratic in Newton's method, once we get half-
        # way to convergence, the next iteration should be exact.
        eps = np.sqrt(precision) / 10.         # /10 is just for extra safety

        # Make sure the initial pq guess is an array copy and uses ab's mask
        pq = Pair(pq_guess.vals.copy(), ab.mask)

        max_dpq = 1.e99
        converged = False
        for count in range(iters):
            ab_test, dab_dpq = PolyFOV._eval_polynomial(pq, coefft,
                                                            dcoefft_p,
                                                            dcoefft_q,
                                                            derivs=False,
                                                            d_dpq=True)

            # Perform one step of Newton's Method
            dpq_dab = dab_dpq.reciprocal(nozeros=True)
                # nozeros=True is safe because dab_dpq can't be zero-valued
            dpq = dpq_dab.chain(ab.wod - ab_test)
            new_max_dpq = dpq.norm().max(builtins=True, masked=-1.)

            if LOGGING.fov_iterations or PolyFOV.DEBUG:
                LOGGING.convergence('PolyFOV._solve_polynomial:',
                                    'iter=%d; change=%.6g' % (count+1,
                                                              new_max_dpq))

            # Quit when convergence stops
            if new_max_dpq <= eps:
                pq += dpq
                converged = True
                break

            if new_max_dpq >= max_dpq:
                break

            pq += dpq.vals
            max_dpq = new_max_dpq

        if not converged:
            LOGGING.warn('PolyFOV._solve_polynomial did not converge;',
                         'iter=%d; change=%.6g' % (count+1, new_max_dpq))

        # Propagate derivatives if necessary
        if derivs:
            new_derivs = {}
            for key, deriv in ab.derivs.items():
                new_derivs[key] = dpq_dab.chain(deriv)

            pq.insert_derivs(new_derivs)

        return pq

################################################################################
# UNIT TESTS
################################################################################

import unittest
import time

class Test_PolyFOV(unittest.TestCase):

    def runTest(self):

        np.random.seed(5294)

        PolyFOV.DEBUG = False
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

        fov = PolyFOV((20,15), coefft_xy_from_uv=coefft_xy_from_uv)

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

        fov = PolyFOV((20,15), coefft_uv_from_xy=coefft_uv_from_xy)

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
