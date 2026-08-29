##########################################################################################
# oops/fov/polynomialfov.py: PolynomialFOV subclass of FOV
##########################################################################################

import numpy as np
import sys

from polymath         import Pair
from oops.config      import LOGGING
from oops.fov         import FOV
from oops.fov.flatfov import FlatFOV

EPSILON = sys.float_info.epsilon/2.         # actual machine precision


class PolynomialFOV(FOV):
    """Subclass of FOV that describes a field of view in which the distortion is described
    by a 2-D polynomial.

    This is the approach used by the Space Telescope Science Institute to describe the
    Hubble instrument fields of view. A PolynomialFOV has no dependence on the optional
    extra indices that can be associated with time, wavelength band, etc.
    """

    DEBUG = False       # Set True to print convergence steps of Newton's Method

    def __init__(self, uv_shape, coefft_xy_from_uv=None,
                 coefft_uv_from_xy=None, uv_los=None, uv_area=None,
                 iters=8, fast=True):
        """Constructor for a PolynomialFOV.

        Parameters:
            uv_shape (float, tuple, or Pair): The size of the field of view in pixels.
                This number can be non-integral if the detector is not composed of a
                rectangular array of pixels.
            coefft_xy_from_uv (array-like, optional): The coefficient array of the
                polynomial to convert `(u,v)` to `(x,y)`. The array has shape
                `(order+1, order+1, 2)`, where `coefft[i,j,0]` is the coefficient on
                `u**i * v**j` yielding `x(u,v)`, and `coefft[i,j,1]` is the coefficient
                yielding `y(u,v)`. If None, then the polynomial for `uv_from_xy` is
                inverted.
            coefft_uv_from_xy (array-like, optional): The coefficient array of the
                polynomial to convert `(x,y)` to `(u,v)`. The array has shape
                `(order+1, order+1, 2)`, where `coefft[i,j,0]` is the coefficient on
                `x**i * y**j` yielding `u(x,y)`, and `coefft[i,j,1]` is the coefficient
                yielding `v(x,y)`. If None, then the polynomial for `xy_from_uv` is
                inverted.
            uv_los (float, tuple, or Pair, optional): The `(u,v)` coordinates of the
                nominal line of sight. By default, this is the midpoint of the rectangle,
                i.e., `uv_shape/2`.
            uv_area (float, optional): The nominal area of a pixel in steradians after
                distortion has been removed.
            iters (int, optional): The number of iterations of Newton's method to use when
                inverting the polynomial.
            fast (bool, optional): If True and both sets of coefficients are provided, the
                polynomials will be used in both directions, meaning that the conversions
                `xy_from_uv` and `uv_from_xy` might be inconsistent, although probably at
                the sub-pixel level. If False, then `uv_from_xy` is refined further using
                one or two steps of Newton's method, which provides consistency at the
                level of machine precision, but `uv_from_xy` will be slightly slower.
        """

        # Prepare coefficients
        self.coefft_xy_from_uv = None
        self.coefft_uv_from_xy = None

        if coefft_xy_from_uv is not None:
            self.coefft_xy_from_uv = np.asarray(coefft_xy_from_uv, dtype=np.float64)
            order = self.coefft_xy_from_uv.shape[0] - 1
            self.coefft_dxy_du = (self.coefft_xy_from_uv[1:] *
                                  np.arange(1,order+1)[:,np.newaxis,np.newaxis])
            self.coefft_dxy_dv = (self.coefft_xy_from_uv[:,1:] *
                                  np.arange(1,order+1)[np.newaxis,:,np.newaxis])

        if coefft_uv_from_xy is not None:
            self.coefft_uv_from_xy = np.asarray(coefft_uv_from_xy, dtype=np.float64)
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

        self.flat_fov = FlatFOV(uv_scale, self.uv_shape, uv_los=self.uv_los)
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
        self.flat_fov = FlatFOV(self.uv_scale, self.uv_shape, uv_los=self.uv_los)

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        # Reference values for precision determinations
        # The goal is full precision in pixel coordinates
        self.uv_precision = EPSILON
        self.xy_precision = EPSILON * min(dx_du, abs(dy_dv))

    def __getstate__(self):
        self.refresh()
        return (self.uv_shape, self.coefft_xy_from_uv, self.coefft_uv_from_xy,
                self.uv_los, self.uv_area, self.iters, self.fast)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by
                PolynomialFOV.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed `(x,y)` coordinates in the camera's frame, with the same
                shape as `uv_pair`.
        """

        # Mask if necessary
        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        if remask:
            uv_pair = uv_pair.remask_or(self.uv_is_outside(uv_pair).vals)

        # Subtract off the center of the field of view
        duv = uv_pair - self.uv_los

        # Transform based on which types of coefficients are given
        if self.coefft_xy_from_uv is not None:
            xy = PolynomialFOV._eval_polynomial(duv,
                                                self.coefft_xy_from_uv,
                                                self.coefft_dxy_du,
                                                self.coefft_dxy_dv,
                                                derivs=derivs)
        else:
            xy_guess = self.flat_fov.xy_from_uv(uv_pair, derivs=False)
            xy = PolynomialFOV._solve_polynomial(duv, xy_guess,
                                                 self.coefft_uv_from_xy,
                                                 self.coefft_duv_dx,
                                                 self.coefft_duv_dy,
                                                 derivs=derivs,
                                                 iters=self.iters,
                                                 precision=self.xy_precision)

        return xy

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates at the
        specified time.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by
                PolynomialFOV.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed `(u,v)` FOV coordinates, with the same shape as `xy_pair`.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)

        # Transform based on which types of coeffs are given
        if self.fast and self.coefft_uv_from_xy is not None:
            duv = PolynomialFOV._eval_polynomial(xy_pair,
                                                 self.coefft_uv_from_xy,
                                                 self.coefft_duv_dx,
                                                 self.coefft_duv_dy,
                                                 derivs=derivs)

        else:
            # If both sets of coefficients are available, use uv_from_xy as the
            # guess. Otherwise, use a flat FOV
            if self.coefft_uv_from_xy is not None:
                duv_guess = PolynomialFOV._eval_polynomial(xy_pair,
                                                           self.coefft_uv_from_xy,
                                                           self.coefft_duv_dx,
                                                           self.coefft_duv_dy,
                                                           derivs=False)
            else:
                duv_guess = (self.flat_fov.uv_from_xy(xy_pair, derivs=False)
                             - self.uv_los)

            # Use the xy_from_uv coefficients to ensure that the polynomial
            # inversion is exact.
            duv = PolynomialFOV._solve_polynomial(xy_pair, duv_guess,
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
            uv = uv.remask_or(self.uv_is_outside(uv).vals)

        return uv

    @staticmethod
    def _eval_polynomial(pq, coefft, dcoefft_p, dcoefft_q, *, derivs=False, d_dpq=False):
        """Evaluate the 2-D polynomial at `(p,q)` to return `(a,b)`.

        Parameters:
            pq (Pair): The points at which to evaluate the polynomial.
            coefft (array-like): The coefficient array defining the polynomial.
            dcoefft_p (array-like): The coefficient array for the polynomial derivative
                with respect to `p`.
            dcoefft_q (array-like): The coefficient array for the polynomial derivative
                with respect to `q`.
            derivs (bool, optional): If True, derivatives are computed and included in the
                returned result.
            d_dpq (bool, optional): If True, the returned quantity is a tuple
                `(ab, dab/dpq)`; otherwise, only `(a,b)` is returned.

        Returns:
            Pair or tuple[Pair, Pair]: Either `(a,b)` or `((a,b), d(a,b)_d(p,q))`,
            depending on the input value of `d_dpq`.

            * `ab` (Pair): The value of the polynomial.
            * `dab_dpq` (Pair): The derivative of `(a,b)` with respect to `(p,q)`.
        """

        pq = Pair.as_pair(pq, recursive=derivs)

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

    @staticmethod
    def _solve_polynomial(ab, pq_guess, coefft, dcoefft_p, dcoefft_q, *, derivs=False,
                          iters=8, precision=0.):
        """Invert the 2-D polynomial to find the `(p,q)` where it evaluates to `(a,b)`.

        Parameters:
            ab (Pair): The value of the polynomial.
            pq_guess (Pair): An initial guess at the `(p,q)` value to return.
            coefft (array-like): The coefficient array defining the polynomial.
            dcoefft_p (array-like): The coefficient array for the polynomial derivative
                with respect to `p`.
            dcoefft_q (array-like): The coefficient array for the polynomial derivative
                with respect to `q`.
            derivs (bool, optional): If True, derivatives are computed and included in the
                returned result.
            iters (int, optional): Maximum number of iterations of Newton's method.
            precision (float, optional): Absolute precision desired. An approximate limit
                is acceptable, and the only down-side of zero (the default) is that the
                solution will require one extra iteration.

        Returns:
            Pair: The `(p,q)` coordinates where the polynomial equals `(a,b)`.
        """

        ab = Pair.as_pair(ab, recursive=derivs)

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
            ab_test, dab_dpq = PolynomialFOV._eval_polynomial(pq, coefft,
                                                              dcoefft_p,
                                                              dcoefft_q,
                                                              derivs=False,
                                                              d_dpq=True)

            # Perform one step of Newton's Method
            dpq_dab = dab_dpq.reciprocal(nozeros=True)
            # nozeros=True is safe because dab_dpq can't be zero-valued
            dpq = dpq_dab.chain(ab.wod - ab_test)
            new_max_dpq = dpq.norm().max(builtins=True, masked=-1.)

            if LOGGING.fov_iterations or PolynomialFOV.DEBUG:
                LOGGING.convergence('PolynomialFOV._solve_polynomial:',
                                    'iter=%d; change=%.6g' % (count+1, new_max_dpq))

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
            LOGGING.warn('PolynomialFOV._solve_polynomial did not converge;',
                         'iter=%d; change=%.6g' % (count+1, new_max_dpq))

        # Propagate derivatives if necessary
        if derivs:
            new_derivs = {}
            for key, deriv in ab.derivs.items():
                new_derivs[key] = dpq_dab.chain(deriv)

            pq.insert_derivs(new_derivs)

        return pq

##########################################################################################
