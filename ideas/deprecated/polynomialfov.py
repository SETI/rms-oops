################################################################################
# oops/fov/polynomialfov.py: PolynomialFOV subclass of FOV
################################################################################

from __future__ import print_function

import numpy as np
from polymath import Pair

from oops.fov import FOV

class PolynomialFOV(FOV):
    """Subclass of FOV that describes a field of view in which the distortion is
    described by a 2-D polynomial.

    This is the approached used by Space Telescope Science Institute to describe
    the Hubble instrument fields of view. A PolynomialFOV has no dependence on
    the optional extra indices that can be associated with time, wavelength
    band, etc.
    """

    DEBUG = False       # True to print(convergence steps on xy_from_uv())

    #===========================================================================
    def __init__(self, uv_shape, coefft_xy_from_uv=None,
                 coefft_uv_from_xy=None, uv_los=None, uv_area=None,
                 iters=8):
        """Constructor for a PolynomialFOV.

        Inputs:
            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            coefft_xy_from_uv
                        the coefficient array of the polynomial to convert U,V
                        to X,Y. The array has shape [order+1,order+1,2], where
                        coefft[i,j,0] is the coefficient on (u**i * v**j)
                        yielding x(u,v), and coefft[i,j,1] is the coefficient
                        yielding y(u,v). All coefficients are 0 for (i+j) >
                        order. If None, then the polynomial for uv_from_xy is
                        inverted.

            coefft_uv_from_xy
                        the coefficient array of the polynomial to convert X,Y
                        to U,V. The array has shape [order+1,order+1,2], where
                        coefft[i,j,0] is the coefficient on (x**i * y**j)
                        yielding u(x,y), and coefft[i,j,1] is the coefficient
                        yielding v(x,y). All coefficients are 0 for (i+j) >
                        order. If None, then the polynomial for xy_from_uv is
                        inverted.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal area of a
                        pixel in steradians after distortion has been removed.

            iters       the number of iterations of Newton's method to use when
                        inverting the polynomial.
        """

        self.coefft_xy_from_uv = None
        self.coefft_uv_from_xy = None

        if coefft_xy_from_uv is not None:
            self.coefft_xy_from_uv = np.asarray(coefft_xy_from_uv)
        if coefft_uv_from_xy is not None:
            self.coefft_uv_from_xy = np.asarray(coefft_uv_from_xy)

        assert (self.coefft_xy_from_uv is not None or
                self.coefft_uv_from_xy is not None)

        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float()
            self.uv_los.as_readonly()

        self.iters = iters

        # Required attribute
        if self.coefft_uv_from_xy is not None:
            self.uv_scale = Pair.as_pair((1./self.coefft_uv_from_xy[1,0,0],
                                          1./self.coefft_uv_from_xy[0,1,1]))
        else:
            self.uv_scale = Pair.as_pair((self.coefft_xy_from_uv[1,0,0],
                                          self.coefft_xy_from_uv[0,1,1]))

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

    def __getstate__(self):
        return (self.uv_shape, self.coefft_xy_from_uv, self.coefft_uv_from_xy,
                self.uv_los, self.uv_area, self.iters)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv, time=None, derivs=False, remask=False,
                                                       fast=False):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv          (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute times. Ignored by
                        PolynomialFOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            fast        If True, a faster, but possibly less robust, convergence
                        criterion is used.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        # Subtract off the center of the field of view
        uv = Pair.as_pair(uv, recursive=derivs) - self.uv_los

        # Transform based on which types of coeffs are given
        if self.coefft_xy_from_uv is not None:
            xy = self._apply_polynomial(uv, self.coefft_xy_from_uv,
                                        derivs=derivs, from_='uv')
        else:
            xy = self._solve_polynomial(uv, self.coefft_uv_from_xy,
                                        derivs=derivs, from_='uv', fast=fast)

        return xy

    #===========================================================================
    def uv_from_xyt(self, xy, time=None, derivs=False, remask=False,
                                                       fast=False):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy          (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute times. Ignored by
                        PolynomialFOV.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            fast        If True, a faster, but possibly less robust, convergence
                        criterion is used.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        xy = Pair.as_pair(xy, recursive=derivs)

        # Transform based on which types of coeffs are given
        if self.coefft_uv_from_xy is not None:
            uv = self._apply_polynomial(xy, self.coefft_uv_from_xy,
                                        derivs=derivs, from_='xy')
        else:
            uv = self._solve_polynomial(xy, self.coefft_xy_from_uv,
                                        derivs=derivs, from_='xy', fast=fast)

        # Add back the center of the field of view
        uv = uv + self.uv_los

        return uv

    #===========================================================================
    def _apply_polynomial(self, pq, coefft, derivs, from_):
        """Apply the polynomial to pair (p,q) to return (a,b).

        Input:
            pq          Pairs of arbitrary shape specifying the points at which
                        to evaluate the polynomial.
            coefft      The coefficient array defining the polynomial.
            derivs      If True, derivatives are computed and included in the
                        result.
            from_       Source system, for labeling the derivatives, e.g., 'uv'
                        or 'xy'.

        Output:         ab
            ab          Pairs of the same shape as pq giving the values of
                        the polynomial at each input point.
        """

        assert from_ in ('uv', 'xy')
        dkey = from_

        order = coefft.shape[0]-1

        (p,q) = pq.to_scalars()
        if pq.shape:
            p = p.vals[..., np.newaxis]
            q = q.vals[..., np.newaxis]
        else:
            p = p.vals
            q = q.vals

        # Construct the powers of line and sample
        p_powers = [1.]
        q_powers = [1.]
        for k in range(1, order + 1):
            p_powers.append(p_powers[-1] * p)
            q_powers.append(q_powers[-1] * q)

        # Evaluate the polynomials
        #
        # Start with the high-order terms and work downward, because this
        # improves accuracy. Stop at one because there are no zero-order terms.
        ab_vals = np.zeros(pq.shape + (2,))
        for k in range(order, -1, -1):
          for i in range(k+1):
            j = k - i
            ab_vals += coefft[i,j,:] * p_powers[i] * q_powers[j]
        ab = Pair(ab_vals, pq.mask)

        # Calculate derivatives if necessary
        if derivs:

            # Compute derivatives
            dab_dpq_vals = np.zeros(pq.shape + (2,2))

            for k in range(order, 0, -1):
              for i in range(k+1):
                j = k - i
                dab_dpq_vals[...,:,0] += (coefft[i,j,:] *
                                          i*p_powers[i-1] * q_powers[j])
                dab_dpq_vals[...,:,1] += (coefft[i,j,:] *
                                          p_powers[i] * j*q_powers[j-1])
            dab_dpq = Pair(dab_dpq_vals, pq.mask, drank=1)

            # Propagate derivatives
#            ab.propagate_deriv(pq, dkey, dab_dpq, derivs)
            new_derivs = {dkey:dab_dpq}
            if pq.derivs:
                for (key, pq_deriv) in pq.derivs.items():
                    new_derivs[key] = dab_dpq.chain(pq_deriv)
            ab.insert_derivs(new_derivs)

        return ab

    #===========================================================================
    def _guess(self, ab, coefft, from_):
        """Compute the initial guess for polynomial inversion.

        Input:
            ab          Pairs of arbitrary shape specifying the points at which
                        to compute the guess.
            coefft      The coefficient array defining the polynomial.
            from_       Source system, for labeling the derivatives, e.g., 'uv'
                        or 'xy'.

        Output:         pq
            pq          Pairs of of the same shape as ab giving the values of
                        the inverted polynomial at each input point.
        """

        if from_ == 'xy':
          return (ab - coefft[0,0]).element_div(self.uv_scale, recursive=False)
        if from_ == 'uv':
          return (ab - coefft[0,0]).element_mul(self.uv_scale, recursive=False)

    #===========================================================================
    def _solve_polynomial(self, ab, coefft, derivs, from_, fast=False):
        """Solve the polynomial for an (a,b) pair to return (p,q).

        Input:
            ab          Pairs of arbitrary shape specifying the points at which
                        to invert the polynomial.
            coefft      The coefficient array defining the polynomial to invert.
            derivs      If True, derivatives are included in the output.
            from_       Source system, for labeling the derivatives, e.g., 'uv'
                        or 'xy'.
            fast        If True, a faster, but possibly less robust, convergence
                        criterion is used.  The unittests with SpeedTest = True
                        produced the folowing results:

                        Slow Newton's method: convergence: 6.68551206589 ms
                        Fast Newton's method: convergence: 5.97627162933 ms
                        Slow/Fast =  1.11867607106

        Output:         pq
            pq          Pairs of of the same shape as ab giving the values of
                        the inverted polynomial at each input point.
        """

        src = {'uv','xy'}
        assert from_ in src
        to_ = (src^{from_}).pop()
        dkey = from_

        ab = Pair.as_pair(ab, recursive=derivs)
        ab_wod = ab.wod

        # Make a rough initial guess
        pq = self._guess(ab_wod, coefft, from_)
        pq.insert_deriv(dkey, Pair.IDENTITY)

        # Iterate until convergence...
        epsilon = 1.e-15
        prev_dpq_max = 1.e99
        for count in range(self.iters):

            # Evaluate the forward transform and its partial derivatives
            ab0 = self._apply_polynomial(pq, coefft, derivs=True, from_=to_)
            dab_dpq = ab0.derivs[dkey]

            # Apply one step of Newton's method in 2-D
            dab = ab_wod - ab0.wod

            dpq_dab = dab_dpq.reciprocal()
            dpq = dpq_dab.chain(dab)
            pq += dpq

            # Convergence tests...

            # simpler, but faster convergence
            if fast:

                # Compare the max step size with the max coordinate value
                # This removes any positional dependence of the relative
                # error.  The denominator can only be zero if the entire
                # grid is (0,0).
                error_max = abs(dpq.vals).max() / abs(pq.vals).max()
                if PolynomialFOV.DEBUG:
                   print(count+1, error_max)

                # Test for root
                #  This eliminates cases where the iteration bounces
                #  around within epsilon of a solution.
                if abs(dab).max() <= epsilon:
                    break

                # Relative correction below epsilon.
                if error_max <= epsilon:
                    break

            # slower convergence, but more robust
            else:

                # The old convergence test below was looking for the correction
                # to overshoot.  This may in principle be a more robust way to
                # ensure machine precision is achieved, but it requires some
                # additonal iterations conmpared to the simpler test above.
                dpq_max = abs(dpq).max()
                if PolynomialFOV.DEBUG:
                    print(iter, dpq_max)

                if dpq_max >= prev_dpq_max:
                    break

            prev_dpq_max = dpq_max

        if PolynomialFOV.DEBUG:
            print(iter+1, 'iterations')

        pq = pq.wod

        # Propagate derivatives if necessary
#        pq.propagate_deriv(ab, dkey, dpq_dab, derivs)
        if derivs:
            new_derivs = {dkey:dpq_dab}
            if ab.derivs:
                for (key, ab_deriv) in ab.derivs.items():
                    new_derivs[key] = dpq_dab.chain(ab_deriv)
            pq.insert_derivs(new_derivs)

        return pq

################################################################################
# UNIT TESTS
################################################################################

import unittest
import time

class Test_PolynomialFOV(unittest.TestCase):

    def runTest(self):

        import time
        from oops.fov.polyfov import PolyFOV

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
            for fast in (True, False):
                t0 = time.time()
                for k in range(iters):
                    xy = fov.xy_from_uv(uv, derivs=True)
                    uv_test = fov.uv_from_xy(xy, derivs=True, fast=fast)
                t1 = time.time()
                print('%s time = %.2f ms' % ('fast' if fast else 'slow',
                                             (t1-t0)/iters*1000.))
        else:
            xy = fov.xy_from_uv(uv, derivs=True)
            uv_test = fov.uv_from_xy(xy, derivs=False)

        self.assertTrue(abs(uv - uv_test).max() < 1.e-14)

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

        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(20,15,2)))
        xy.insert_deriv('rs', Pair(np.random.randn(20,15,2,2), drank=1))
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

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

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

        fov = PolynomialFOV((20,15), coefft_uv_from_xy=coefft_uv_from_xy,
                                     uv_los=(7,7), uv_area=1.)

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

        dxy_dt = dxy_du * uv.d_dt.vals[...,0] + dxy_dv * uv.d_dt.vals[...,1]
        dxy_dr = dxy_du * uv.d_drs.vals[...,0,0] + dxy_dv * uv.d_drs.vals[...,1,0]
        dxy_ds = dxy_du * uv.d_drs.vals[...,0,1] + dxy_dv * uv.d_drs.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
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
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,0] - dxy_dr.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_drs.vals[...,1] - dxy_ds.vals).max() <= DEL)

        ########################################
        # Only xy_from_uv defined, comparison to PolyFOV
        ########################################

        coefft_xy_from_uv = np.zeros((3,3,2))
        coefft_xy_from_uv[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                             [ 1.20, -0.01,  0.00],
                                             [-0.02,  0.00,  0.00]])
        coefft_xy_from_uv[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                             [-0.20, -0.03,  0.00],
                                             [-0.02,  0.00,  0.00]])

        fov = PolynomialFOV((20,15), coefft_xy_from_uv=coefft_xy_from_uv)
        polyfov = PolyFOV((20,15), coefft_xy_from_uv=coefft_xy_from_uv)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(0,1648,20), np.arange(0,129,8))
        uv.insert_deriv('t' , Pair(np.random.randn(83,17,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(83,17,2,2), drank=1))

        xy1 = fov.xy_from_uv(uv, derivs=True)
        xy2 = polyfov.xy_from_uv(uv, derivs=True)

        DEL = 1.e-11
        self.assertTrue(abs(xy1.vals       - xy2.vals      ).max() <= DEL)
        self.assertTrue(abs(xy1.d_dt.vals  - xy2.d_dt.vals ).max() <= DEL)
        self.assertTrue(abs(xy1.d_drs.vals - xy2.d_drs.vals).max() <= DEL)

        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(83,17,2)))
        xy.insert_deriv('rs', Pair(np.random.randn(83,17,2,2), drank=1))

# Failure of PolynomialFOV
#         uv1 = fov.uv_from_xy(xy, derivs=True)
#         uv2 = polyfov.uv_from_xy(xy, derivs=True)
#
#         print(abs(uv1.vals       - uv2.vals      ).max())
#         print(abs(uv1.d_dt.vals  - uv2.d_dt.vals ).max())
#         print(abs(uv1.d_drs.vals - uv2.d_drs.vals).max())
# 7737.898863616556
# 10.64042327053394
# 20.83167707967607
#
#         DEL = 3.e-12
#         self.assertTrue(abs(uv1.vals       - uv2.vals      ).max() <= DEL)
#         self.assertTrue(abs(uv1.d_dt.vals  - uv2.d_dt.vals ).max() <= DEL)
#         self.assertTrue(abs(uv1.d_drs.vals - uv2.d_drs.vals).max() <= DEL)

        ########################################
        # Only uv_from_xy defined, comparison to PolyFOV
        ########################################

        coefft_uv_from_xy = np.zeros((3,3,2))
        coefft_uv_from_xy[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                             [ 1.20, -0.01,  0.00],
                                             [-0.02,  0.00,  0.00]])
        coefft_uv_from_xy[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                             [-0.20, -0.03,  0.00],
                                             [-0.02,  0.00,  0.00]])

        fov = PolynomialFOV((20,15), coefft_uv_from_xy=coefft_uv_from_xy)
        polyfov = PolyFOV((20,15), coefft_uv_from_xy=coefft_uv_from_xy)

        #### uv -> xy -> uv, with derivs

        uv = Pair.combos(np.arange(0,1648,20), np.arange(0,129,8))
        uv.insert_deriv('t' , Pair(np.random.randn(83,17,2)))
        uv.insert_deriv('rs', Pair(np.random.randn(83,17,2,2), drank=1))

# Failure of PolynomialFOV
#         xy1 = fov.xy_from_uv(uv, derivs=True)
#         xy2 = polyfov.xy_from_uv(uv, derivs=True)
#
#         print(abs(xy1.vals       - xy2.vals      ).max())
#         print(abs(xy1.d_dt.vals  - xy2.d_dt.vals ).max())
#         print(abs(xy1.d_drs.vals - xy2.d_drs.vals).max())
# 64902.23479578221
# 960.3106247521578
# 1099.824786459409
#
#         DEL = 1.e-11
#         self.assertTrue(abs(xy1.vals       - xy2.vals      ).max() <= DEL)
#         self.assertTrue(abs(xy1.d_dt.vals  - xy2.d_dt.vals ).max() <= DEL)
#         self.assertTrue(abs(xy1.d_drs.vals - xy2.d_drs.vals).max() <= DEL)

        #### xy -> uv -> xy, with derivs

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t' , Pair(np.random.randn(83,17,2)))
        xy.insert_deriv('rs', Pair(np.random.randn(83,17,2,2), drank=1))

        uv1 = fov.uv_from_xy(xy, derivs=True)
        uv2 = polyfov.uv_from_xy(xy, derivs=True)

        DEL = 3.e-12
        self.assertTrue(abs(uv1.vals       - uv2.vals      ).max() <= 2.e-8)
        self.assertTrue(abs(uv1.d_dt.vals  - uv2.d_dt.vals ).max() <= 3.e-13)
        self.assertTrue(abs(uv1.d_drs.vals - uv2.d_drs.vals).max() <= 3.e-13)

        xy.insert_deriv('t', Pair((1,1)))
        uv_test = fov.uv_from_xy(xy, derivs=False)
        self.assertEqual(uv_test.derivs, {})

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
