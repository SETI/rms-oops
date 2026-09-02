##########################################################################################
# oops/fov/barrelfov.py
##########################################################################################

import sys

import numpy as np

from polymath         import Scalar, Pair
from oops.config      import LOGGING
from oops.fov         import FOV
from oops.fov.flatfov import FlatFOV

EPSILON = sys.float_info.epsilon / 2.   # actual machine precision


class BarrelFOV(FOV):
    """Subclass of FOV that describes a field of view in which the distortion is described
    by a 1-D polynomial in distance from the image center.
    """

    # True to print convergence steps in _solve_ratio()
    DEBUG = False

    def __init__(self, uv_scale, uv_shape, *, coefft_xy_from_uv=None,
                 coefft_uv_from_xy=None, uv_los=None, uv_area=None, iters=8, fast=True):
        """Constructor for a BarrelFOV.

        Parameters:
            uv_scale (float, tuple, or Pair): The ratios `dx/du` and `dy/dv` at the
                center of the FOV. For example, if `(u,v)` are in units of arcseconds,
                then::

                    uv_scale = Pair((pi/180/3600.,pi/180/3600.))

                Use the sign of the second element to define the direction of increasing
                `v`: negative for up, positive for down.
            uv_shape (tuple, Pair, int, or float): The size of the field of view in
                pixels. This number can be non-integral if the detector is not composed of
                a rectangular array of pixels.
            coefft_xy_from_uv (ndarray, optional): The polynomial coefficient array
                describing the radial distortion from `(u,v)` to `(x,y)`. It is a function
                of `r`, defined as::

                    r = sqrt(((u-uv_los[0]) * uv_scale[0])**2 +
                             ((v-uv_los[1]) * uv_scale[1])**2)

                In other words, `r` is in units of radians and measures the distance from
                the center of the FOV if there were no distortion. The polynomial `f(r)`
                returns the distorted distance given the un-distorted distance. Because
                this polynomial cannot have a constant term, the coefficients begin with
                the linear term, which is typically ~ 1. In other words,
                `coefft_xy_from_uv[i]` is the coefficient on `r**(i+1)`. If this input is
                None, the distortion polynomial for `uv_from_xy` is inverted.
            coefft_uv_from_xy (ndarray, optional): The polynomial coefficient array
                describing the radial distortion scale factor from `(x,y)` to `(u,v)`. It
                is a function of `r`, defined as::

                    r = sqrt(x**2 + y**2),

                in units of radians. The array has shape `(order,)` under the assumption
                that there can be no constant term, so `coefft_uv_from_xy[i]` is the
                coefficient on `r**(i+1)`. The first coefficient is typically ~ 1,
                implying no distortion at the center of the FOV. If None, the distortion
                polynomial for `xy_from_uv` is inverted.
            uv_los (float, tuple, or Pair, optional): The `(u,v)` coordinates of the
                nominal line of sight. By default, this is the midpoint of the rectangle,
                i.e., `uv_shape/2`.
            uv_area (float, optional): The nominal area of a pixel in steradians after
                distortion has been removed.
            iters (int, optional): The number of iterations of Newton's method to use when
                inverting the distortion polynomial.
            fast (bool, optional): If True and both sets of coefficients are provided, the
                polynomials will be used in both directions, meaning that the conversions
                `xy_from_uv` and `uv_from_xy` might be inconsistent, although probably at
                the sub-pixel level. If False, then `uv_from_xy` is refined further using
                one or two steps of Newton's method, which provides consistency at the
                level of machine precision, but `uv_from_xy` will be somewhat slower.
        """

        self.coefft_xy_from_uv = None
        self.coefft_uv_from_xy = None

        # Save the coefficients
        #
        # The function we evaluate is actually polynomial(r)/r, which is very well
        # behaved (nearly constant) in both directions.

        if coefft_xy_from_uv is not None:
            order = len(coefft_xy_from_uv)
            self.coefft_xy_from_uv = np.asarray(coefft_xy_from_uv, dtype=np.float64)
            self.dcoefft_xy_from_uv = self.coefft_xy_from_uv * np.arange(order)

        if coefft_uv_from_xy is not None:
            order = len(coefft_uv_from_xy)
            self.coefft_uv_from_xy = np.asarray(coefft_uv_from_xy, dtype=np.float64)
            self.dcoefft_uv_from_xy = self.coefft_uv_from_xy * np.arange(order)

        if (self.coefft_xy_from_uv is None and
            self.coefft_uv_from_xy is None):
                raise ValueError('at least one of coefft_xy_from_uv and '
                                 'coefft_uv_from_xy must be specified')

        self.uv_scale = Pair.as_pair(uv_scale).as_readonly()
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float()
            self.uv_los.as_readonly()

        self.iters = max(int(iters), 2)
        self.fast = bool(fast) and (self.coefft_uv_from_xy is not None)

        self.flat_fov = FlatFOV(self.uv_scale, self.uv_shape, uv_los=self.uv_los)

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        # Reference values for precision determinations
        # The goal is full precision in pixel coordinates
        self.uv_precision = EPSILON
        self.xy_precision = EPSILON * np.min(np.abs(self.uv_scale.vals))

    def __getstate__(self):
        self.refresh()
        return (self.uv_scale, self.uv_shape, self.coefft_xy_from_uv,
                self.coefft_uv_from_xy, self.uv_los, self.uv_area, self.iters,
                self.fast)

    def __setstate__(self, state):
        (uv_scale, uv_shape, coefft_xy_from_uv, coefft_uv_from_xy, uv_los, uv_area,
         iters, fast) = state
        self.__init__(uv_scale, uv_shape, coefft_xy_from_uv=coefft_xy_from_uv,
                      coefft_uv_from_xy=coefft_uv_from_xy, uv_los=uv_los,
                      uv_area=uv_area, iters=iters, fast=fast)
        self.freeze()

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by BarrelFOV.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed `(x,y)` coordinates in the camera's frame.
        """

        # Convert to xy using flat FOV model
        flat_xy = self.flat_fov.xy_from_uv(uv_pair, derivs=derivs, remask=remask)
        r_flat = flat_xy.norm(recursive=derivs)

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

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates at the
        specified time.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by BarrelFOV.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed `(u,v)` FOV coordinates, with the same shape as `xy_pair`.
        """

        true_xy = Pair.as_pair(xy_pair, recursive=derivs)
        r_true = true_xy.norm(recursive=derivs)

        # Distort based on which types of coefficients are given; the polynomial must be
        # evaluated directly if there is no xy_from_uv polynomial to invert
        if self.fast or self.coefft_xy_from_uv is None:
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

    @staticmethod
    def _eval_ratio(r, coefft, dcoefft, *, derivs=False, d_dr=False):
        """Compute the ratio polynomial(r) / r.

        By returning the ratio instead of the polynomial value directly, it is easier to
        handle `r = polynomial(r) = 0`.

        Parameters:
            r (Scalar): The points at which to evaluate the polynomial.
            coefft (ndarray): The coefficient array defining the polynomial, with the
                leading zero-valued constant term omitted.
            dcoefft (ndarray): The coefficients of the derivative of the ratio, i.e.,
                `coefft * [0,1,2,...]`.
            derivs (bool, optional): True to include the derivatives embedded in `r` in
                the result.
            d_dr (bool, optional): If True, the returned quantity is a tuple
                `(ratio, dratio/dr)`; otherwise, only `ratio` is returned.

        Returns:
            Scalar or tuple[Scalar, Scalar]: Either `ratio` or `(ratio, dratio_dr)`,
            depending on the input value of `d_dr`.

            * `ratio` (Scalar): The value of `polynomial(r) / r`.
            * `dratio_dr` (Scalar): The derivative of `ratio` with respect to `r`.
        """

        # Construct the powers of radius, starting at 1
        r = Scalar.as_scalar(r, recursive=derivs)

        powers = np.empty(r.shape + coefft.shape)
        powers[...,0] = 1.
        powers[...,1] = r.vals
        for k in range(2, coefft.shape[0]):
            powers[...,k] = powers[...,k-1] * r.vals

        # Evaluate the polynomial
        ratio = Scalar(np.sum(powers * coefft, axis=-1), r.mask)

        # Evaluate the derivative with respect to r if necessary
        # Note that dcoefft[0] is always 0.
        if d_dr or derivs:
            dratio_dr = Scalar(np.sum(dcoefft[1:] * powers[...,:-1], axis=-1))
            # unmasked is OK

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

    @staticmethod
    def _solve_ratio(f, r_guess, coefft, dcoefft, *, derivs=False, iters=8, precision=0.):
        """Invert a 1-D polynomial to find `r` where `polynomial(r) = f`, returning `r/f`.

        Using the ratio `r/f` instead of `r` itself makes it easier to handle `r = f = 0`.

        Parameters:
            f (Scalar): The values of the polynomial.
            r_guess (Scalar): Initial guess at the values of `r`.
            coefft (ndarray): Coefficient array defining the polynomial, with the leading
                zero-valued constant term omitted.
            dcoefft (ndarray): The coefficients of the derivative of the ratio
                `polynomial(r) / r`, i.e., `coefft * [0,1,2,...]`.
            derivs (bool, optional): True to include the derivatives embedded in `f` in
                the result.
            iters (int, optional): The maximum number of iterations of Newton's method.
            precision (float, optional): Absolute precision desired. An approximate limit
                is OK, and the only down-side of zero (the default) is that the solution
                will require one extra iteration.

        Returns:
            Scalar: The ratio `r/f`, where `r` is the value at which the polynomial
            evaluates to `f`.
        """

        f = Scalar.as_scalar(f, recursive=derivs)

        # Handle fully-masked case
        if np.all(f.mask):
            return Scalar(np.ones(f.shape), True)

        # Because convergence is quadratic in Newton's method, once we get half- way to
        # convergence, the next iteration should be exact.
        eps = 2*[precision * 2] + (iters-2) * [np.sqrt(precision) / 30]
            # Don't assume the convergence is quadratic till the third iteration
            # Division by 30 is just for extra safety

        # Make sure the initial r guess is an array copy and uses f's mask
        r = r_guess.copy().remask(f.mask)

        max_dr = 1.e99
        converged = False
        for count in range(iters):
            (f_over_r, d_f_over_r_dr) = BarrelFOV._eval_ratio(r, coefft, dcoefft,
                                                              derivs=False, d_dr=True)
            f_test = f_over_r * r
            df_dr = f_over_r + r * d_f_over_r_dr

            # Perform one step of Newton's Method
            dr = (f.wod - f_test) / df_dr
                # Note that df_dr should never be zero, so this is safe
            new_max_dr = abs(dr).max(builtins=True, masked=-1.)

            if LOGGING.fov_iterations or BarrelFOV.DEBUG:
                LOGGING.convergence('BarrelFOV._solve_ratio:',
                                    'iter=%d; change=%.6g' % (count+1, new_max_dr))

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
                         'iter=%d; change=%.6g' % (count+1, new_max_dr))

        # Prepare ratio r/f
        ratio = 1. / f_over_r   # f_over_r can't be zero

        # Propagate derivatives if necessary
        if derivs:
            new_derivs = {}
            for key, df_dx in f.derivs.items():

                # We need to obtain dratio_dx while avoiding divide-by-zero

                dr_dx = df_dx / df_dr   # df_dr cannot equal zero

                # d(ratio)/dx = d(1/f_over_r)/dx
                #   = -d(f_over_r)/dx / f_over_r**2
                #   = -d(f_over_r)/dr * dr/dx / f_over_r**2
                new_derivs[key] = -d_f_over_r_dr * dr_dx * ratio**2

            ratio.insert_derivs(new_derivs)

        return ratio

##########################################################################################
