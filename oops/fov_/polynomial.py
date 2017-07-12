################################################################################
# oops/fov_/polynomial.py: Polynomial subclass of FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class Polynomial(FOV):
    """The Polynomial subclass of FOV describes a field of view in which the
    distortion is described by a 2-D polynomial. This is the approached used by
    Space Telescope Science Institute to describe the Hubble instrument fields
    of view. A Polynomial FOV has no dependence on the optional extra indices
    that can be associated with time, wavelength band, etc.
    """

    DEBUG = False       # True to print convergence steps on xy_from_uv()

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
                        the coefficient array of the polynomial to
                        convert U,V to X,Y. The array has shape
                        [order+1,order+1,2], where coefft[i,j,0] is the
                        coefficient on (u**i * v**j) yielding x(u,v), and
                        coefft[i,j,1] is the coefficient yielding y(u,v). All
                        coefficients are 0 for (i+j) > order. If None, then
                        the polynomial for uv_from_xy is inverted.

            coefft_uv_from_xy
                        the coefficient array of the polynomial to
                        convert X,Y to U,V. The array has shape
                        [order+1,order+1,2], where coefft[i,j,0] is the
                        coefficient on (x**i * y**j) yielding u(x,y), and
                        coefft[i,j,1] is the coefficient yielding v(x,y). All
                        coefficients are 0 for (i+j) > order. If None, then
                        the polynomial for xy_from_uv is inverted. 

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal area of a
                        pixel in steradians after distortion has been removed.

            uv_from_xy    True to indicate that the polynomial converts from
                        (x,y) to (u,v). The default is that it converts from
                        (u,v) to (x,y).
                        
            iters       the number of iterations of Newton's method to use when
                        evaluating uv_from_xy().
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

    ########################################

    def xy_from_uv(self, uv_pair, derivs=False):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.
        """

        # Subtract off the center of the field of view
        uv_pair = Pair.as_pair(uv_pair, derivs) - self.uv_los
        
        if self.coefft_xy_from_uv is not None:
            xy = self._apply_polynomial(uv_pair, self.coefft_xy_from_uv, derivs)
        else:
            xy = self._solve_polynomial(uv_pair, self.coefft_uv_from_xy, derivs,
                                        True)

        return xy
        
    def uv_from_xy(self, xy_pair, derivs=False):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.
        """

        xy_pair = Pair.as_pair(xy_pair, derivs)
        
        if self.coefft_uv_from_xy is not None:
            uv = self._apply_polynomial(xy_pair, self.coefft_uv_from_xy, derivs)
        else:
            uv = self._solve_polynomial(xy_pair, self.coefft_xy_from_uv, derivs,
                                        False)

        uv = uv + self.uv_los
        
        return uv

    def _apply_polynomial(self, uv_pair, coefft, derivs):
        """Apply the polynomial to a (u,v) pair to return (x,y).
        
        The terminology in this function assumes that the polynomial takes
        (u,v) and yields (x,y). However, there is no assumption as to what
        (u,v) and (x,y) actually represent, so this routine can be called
        with (u,v) and (x,y) swapped from the point of view of the FOV.
        """
        
        order = coefft.shape[0]-1
        
        (du,dv) = uv_pair.to_scalars()
        du = du.vals[..., np.newaxis]
        dv = dv.vals[..., np.newaxis]

        # Construct the powers of line and sample
        du_powers = [1.]
        dv_powers = [1.]
        for k in range(1, order + 1):
            du_powers.append(du_powers[-1] * du)
            dv_powers.append(dv_powers[-1] * dv)

        # Initialize the output
        xy_pair_vals = np.zeros(uv_pair.shape + (2,))

        # Evaluate the polynomials
        #
        # Start with the high-order terms and work downward, because this
        # improves accuracy. Stop at one because there are no zero-order terms.
        for k in range(order, -1, -1):
          for i in range(k+1):
            j = k - i
            xy_pair_vals += coefft[i,j,:] * du_powers[i] * dv_powers[j]

        xy = Pair(xy_pair_vals, uv_pair.mask)

        # Calculate and return derivatives if necessary
        if uv_pair.derivs:
            dxy_duv_vals = np.zeros(uv_pair.shape + (2,2))

            for k in range(order, 0, -1):
              for i in range(k+1):
                j = k - i
                dxy_duv_vals[...,:,0] += (coefft[i,j,:] *
                                          i*du_powers[i-1] * dv_powers[j])
                dxy_duv_vals[...,:,1] += (coefft[i,j,:] *
                                          du_powers[i] * j*dv_powers[j-1])

            dxy_duv = Pair(dxy_duv_vals, uv_pair.mask, drank=1)

            new_derivs = {}
            for (key, uv_deriv) in uv_pair.derivs.iteritems():
                new_derivs[key] = dxy_duv.chain(uv_deriv)

            xy.insert_derivs(new_derivs)

        return xy

    def _solve_polynomial(self, xy_pair, coefft, derivs, uv_from_xy):
        """Solve the polynomial for an (x,y) pair to return (u,v).
        
        The terminology in this function assumes that the polynomial takes
        (u,v) and yields (x,y). However, there is no assumption as to what
        (u,v) and (x,y) actually represent, so this routine can be called
        with (u,v) and (x,y) swapped from the point of view of the FOV.
        """

        order = coefft.shape[0]-1

        xy_pair = Pair.as_pair(xy_pair, derivs)
        xy_wod = xy_pair.without_derivs()

        # Make a rough initial guess
        if uv_from_xy:
            uv = (xy_wod - coefft[0,0]).element_mul(self.uv_scale)
        else:
            uv = (xy_wod - coefft[0,0]).element_div(self.uv_scale)
        uv.insert_deriv('uv', Pair.IDENTITY)

        # Iterate until convergence...
        prev_duv_max = 1.e99
        for iter in range(self.iters):

            # Evaluate the transform and its partial derivatives
            xy = self._apply_polynomial(uv, coefft, derivs=True)
            dxy_duv = xy.d_duv

            # Apply one step of Newton's method in 2-D
            dxy = xy_wod - xy.without_derivs()

            duv_dxy = dxy_duv.reciprocal()
            duv = duv_dxy.chain(dxy)
            uv += duv

            # Test for convergence
            duv_max = abs(duv).max()
            if Polynomial.DEBUG:
                print iter, duv_max

            if duv_max >= prev_duv_max: break

            prev_duv_max = duv_max

        uv = uv.without_derivs()

        # Fill in derivatives if necessary
        if xy_pair.derivs:
            new_derivs = {}
            for (key, xy_deriv) in xy_pair.derivs.iteritems():
                new_derivs[key] = duv_dxy.chain(xy_deriv)

            uv.insert_derivs(new_derivs)

        return uv

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Polynomial(unittest.TestCase):

    def runTest(self):

        Polynomial.DEBUG = False

        ############################
        order = 2
        uv_coefft = np.zeros((3,3,2))
        uv_coefft[...,0] = np.array([[ 5.000, -0.100, -0.001],
                                     [ 1.020, -0.001,  0.000],
                                     [-0.002,  0.000,  0.000]])
        uv_coefft[...,1] = np.array([[ 0.000, -1.010,  0.001],
                                     [-0.020, -0.003,  0.000],
                                     [-0.002,  0.000,  0.000]])

        uv = Pair.combos(np.arange(20), np.arange(15))

        fov = Polynomial(uv.shape, coefft_xy_from_uv=uv_coefft, uv_los=(7,7), uv_area=1.)

        xy = fov.xy_from_uv(uv)
        uv_test = fov.uv_from_xy(xy)

        self.assertTrue(abs(uv - uv_test).max() < 1.e-14)

        ############################
        order = 2
        uv_coefft = np.zeros((3,3,2))
        uv_coefft[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                     [ 1.20, -0.01,  0.00],
                                     [-0.02,  0.00,  0.00]])
        uv_coefft[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                     [-0.20, -0.03,  0.00],
                                     [-0.02,  0.00,  0.00]])

        uv = Pair.combos(np.arange(20), np.arange(15))
        uv.insert_deriv('t', Pair((1,1)))

        fov = Polynomial(uv.shape, coefft_xy_from_uv=uv_coefft, uv_los=(7,7), uv_area=1.)

        xy = fov.xy_from_uv(uv, derivs=False)
        self.assertEqual(xy.derivs, {})

        xy.insert_deriv('t', Pair((1,1)))
        uv_test = fov.uv_from_xy(xy, derivs=False)
        self.assertEqual(uv_test.derivs, {})

        self.assertTrue(abs(uv - uv_test).max() < 1.e-14)

        ############################
        order = 2
        uv_coefft = np.zeros((3,3,2))
        uv_coefft[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                     [ 1.20, -0.01,  0.00],
                                     [-0.02,  0.00,  0.00]])
        uv_coefft[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                     [-0.20, -0.03,  0.00],
                                     [-0.02,  0.00,  0.00]])

        uv = Pair.combos(np.arange(20), np.arange(15))
        uv.insert_deriv('t', Pair(np.random.randn(20,15,2)))
        uv.insert_deriv('ab', Pair(np.random.randn(20,15,2,2), drank=1))

        fov = Polynomial(uv.shape, coefft_xy_from_uv=uv_coefft, uv_los=(7,7), uv_area=1.)

        xy = fov.xy_from_uv(uv, derivs=True)

        EPS = 1.e-5
        xy0 = fov.xy_from_uv(uv + (-EPS,0), False)
        xy1 = fov.xy_from_uv(uv + ( EPS,0), False)
        dxy_du = (xy1 - xy0) / (2. * EPS)

        xy0 = fov.xy_from_uv(uv + (0,-EPS), False)
        xy1 = fov.xy_from_uv(uv + (0, EPS), False)
        dxy_dv = (xy1 - xy0) / (2. * EPS)

        dxy_dt = dxy_du * uv.d_dt.vals[...,0] + dxy_dv * uv.d_dt.vals[...,1]
        dxy_da = dxy_du * uv.d_dab.vals[...,0,0] + dxy_dv * uv.d_dab.vals[...,1,0]
        dxy_db = dxy_du * uv.d_dab.vals[...,0,1] + dxy_dv * uv.d_dab.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(xy.d_dt.vals - dxy_dt.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_dab.vals[...,0] - dxy_da.vals).max() <= DEL)
        self.assertTrue(abs(xy.d_dab.vals[...,1] - dxy_db.vals).max() <= DEL)

        ############################
        order = 2
        uv_coefft = np.zeros((3,3,2))
        uv_coefft[...,0] = np.array([[ 5.00, -0.10, -0.01],
                                     [ 1.20, -0.01,  0.00],
                                     [-0.02,  0.00,  0.00]])
        uv_coefft[...,1] = np.array([[ 0.00, -1.10,  0.01],
                                     [-0.20, -0.03,  0.00],
                                     [-0.02,  0.00,  0.00]])

        uv = Pair.combos(np.arange(20), np.arange(15))
        fov = Polynomial(uv.shape, coefft_xy_from_uv=uv_coefft, uv_los=(7,7), uv_area=1.)

        xy = fov.xy_from_uv(uv, derivs=False)
        xy.insert_deriv('t', Pair(np.random.randn(20,15,2)))
        xy.insert_deriv('ab', Pair(np.random.randn(20,15,2,2), drank=1))

        uv = fov.uv_from_xy(xy, derivs=True)

        EPS = 1.e-5
        uv0 = fov.uv_from_xy(xy + (-EPS,0), False)
        uv1 = fov.uv_from_xy(xy + ( EPS,0), False)
        duv_dx = (uv1 - uv0) / (2. * EPS)

        uv0 = fov.uv_from_xy(xy + (0,-EPS), False)
        uv1 = fov.uv_from_xy(xy + (0, EPS), False)
        duv_dy = (uv1 - uv0) / (2. * EPS)

        duv_dt = duv_dx * xy.d_dt.vals[...,0] + duv_dy * xy.d_dt.vals[...,1]
        duv_da = duv_dx * xy.d_dab.vals[...,0,0] + duv_dy * xy.d_dab.vals[...,1,0]
        duv_db = duv_dx * xy.d_dab.vals[...,0,1] + duv_dy * xy.d_dab.vals[...,1,1]

        DEL = 1.e-6
        self.assertTrue(abs(uv.d_dt.vals - duv_dt.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_dab.vals[...,0] - duv_da.vals).max() <= DEL)
        self.assertTrue(abs(uv.d_dab.vals[...,1] - duv_db.vals).max() <= DEL)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
