################################################################################
# oops/fov_/radial.py: Radial subclass of FOV
################################################################################

from __future__ import print_function

from IPython import embed

import numpy as np
import oops

from polymath import *
from oops.fov_.fov import FOV

#******************************************************************************
# Radial FOV class
#******************************************************************************
class Radial(FOV):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """The Radial subclass of FOV describes a field of view in which the
    distortion is described by a 1-D polynomial in distance from the image 
    center.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    DEBUG = False       # True to print(convergence steps on xy_from_uv())

    PACKRAT_ARGS = ['uv_scale', 'uv_shape', 'coefft_xy_from_uv', 'coefft_uv_from_xy',
                    'uv_los', 'uv_area', 'iters']

    #=========================================================================
    # __init__
    #=========================================================================
    def __init__(self, uv_scale, uv_shape, coefft_xy_from_uv=None,
                 coefft_uv_from_xy=None, uv_los=None, uv_area=None,
                 iters=8):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Constructor for a RadialFOV.

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
                        the coefficient array of the polynomial to convert 
                        U,V to X,Y. The array has shape [order+1], where 
                        coefft[i] is the coefficient on r**i, where 
                        r = sqrt((u-ulos)**2 + (v-vlos)**2), yielding x(u,v) 
			and y(u,v). All coefficients are 0 for i > order. If 
			None, then the polynomial for uv_from_xy is inverted.

            coefft_uv_from_xy
                        the coefficient array of the polynomial to convert 
                        X,Y to U,V. The array has shape [order+1], where 
                        coefft[i] is the coefficient on r**i, where 
                        r = sqrt((u-ulos)**2 + (v-vlos)**2), yielding x(u,v) 
			and y(u,v). All coefficients are 0 for i > order. If 
			None, then the polynomial for xy_from_uv is inverted.

###note this include the camera scale unlike in OMINAS

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal area of a
                        pixel in steradians after distortion has been removed.

            iters       the number of iterations of Newton's method to use when
                        inverting the distortion polynomial.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.coefft_xy_from_uv = None
        self.coefft_uv_from_xy = None
        
        if coefft_xy_from_uv is not None:
            self.coefft_xy_from_uv = np.asarray(coefft_xy_from_uv)
        if coefft_uv_from_xy is not None:
            self.coefft_uv_from_xy = np.asarray(coefft_uv_from_xy)

        assert (self.coefft_xy_from_uv is not None or
                self.coefft_uv_from_xy is not None)

        self.uv_scale = Pair.as_pair(uv_scale).as_readonly()
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float()
            self.uv_los.as_readonly()

        self.iters = iters
	
        self.flat_fov = \
                   oops.fov.FlatFOV(self.uv_scale, self.uv_shape, self.uv_los)


        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area
    #=========================================================================



    #=========================================================================
    # xy_from_uv
    #=========================================================================
    def xy_from_uv(self, uv, derivs=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        if self.coefft_xy_from_uv is not None:
            xy = self._forward_polynomial(uv, 
                       self.coefft_xy_from_uv, self.flat_fov.xy_from_uv, derivs)
        else:
            xy = self._inverse_polynomial(uv, 
                       self.coefft_uv_from_xy, self.flat_fov.xy_from_uv, derivs)

        return xy
    #=========================================================================



    #=========================================================================
    # uv_from_xy
    #=========================================================================
    def uv_from_xy(self, xy, derivs=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        xy = Pair.as_pair(xy, derivs)

        if self.coefft_uv_from_xy is not None:
            uv = self._forward_polynomial(xy, 
                       self.coefft_uv_from_xy, self.flat_fov.uv_from_xy, derivs)
        else:
            uv = self._inverse_polynomial(xy, 
                       self.coefft_xy_from_uv, self.flat_fov.uv_from_xy, derivs)
        
        return uv
    #=========================================================================



    #=========================================================================
    # _compute_polynomial
    #=========================================================================
    def _compute_polynomial(self, r, coefft, derivs=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Compute a polynomial in one variable 
           (this should be a library function)..
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        order = coefft.shape[0]-1
        
        #------------------------------------
	# Construct the powers of radius
        #------------------------------------
        r_powers = [1.]
        for k in range(1, order + 1):
	    r_powers.append(r_powers[-1] * r)

        #-----------------------------------------------------------------
        # Evaluate the polynomial
        #
        # Start with the high-order terms and work downward, because this
        # improves accuracy. Stop at one because there are no zero-order 
        # terms.
        #-----------------------------------------------------------------
        result = 0.
        for i in range(order, -1, -1):
            result += coefft[i] * r_powers[i]

        #---------------------------------------------------
        # Calculate and return derivatives if necessary
        #---------------------------------------------------
        deriv = None
        if derivs:
            deriv = 0.
            for i in range(order, -1, -1):
                deriv += i*coefft[i]*r_powers[i-1]
		

        return (result, deriv)
    #=========================================================================



    #=========================================================================
    # _forward_polynomial
    #=========================================================================
    def _forward_polynomial(self, ab, coefft, flat, derivs):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Apply the polynomial to a pair (a,b) to return (p,q).
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #------------------------------------
	# Compute radii
        #------------------------------------
        a = ab.vals[...,0]
        b = ab.vals[...,1]
        r = ab.norm_sq().vals
	r[np.where(r==0)] = 1  ########################

        #------------------------------------
	# Compute the polynomial correction
        #------------------------------------
        (cor, dcor) = self._compute_polynomial(r, coefft, derivs=derivs)
        pq = flat(ab, derivs=derivs)*cor

        #---------------------------------------------------
        # Calculate and return derivatives if necessary
        #---------------------------------------------------
        if ab.derivs:
            dpq_dab_vals = np.zeros(ab.shape + (2,2))

            dp_da = cor + dcor * a**2/r
            dq_db = cor + dcor * b**2/r
            dq_da = dcor * a/r
            dp_db = dcor * b/r

            dpq_dab_vals[...,0,0] = np.squeeze(dp_da)
            dpq_dab_vals[...,1,1] = np.squeeze(dq_db)
            dpq_dab_vals[...,0,1] = np.squeeze(dp_db)
            dpq_dab_vals[...,1,0] = np.squeeze(dq_da)

            dpq_dab = Pair(dpq_dab_vals, ab.mask, drank=1)

            new_derivs = {}
            for (key, pq_deriv) in ab.derivs.items():
                new_derivs[key] = dpq_dab.chain(pq_deriv)

            pq.insert_derivs(new_derivs)

        return pq
    #=========================================================================



    #=========================================================================
    # _inverse_polynomial
    #=========================================================================
    def __inverse_polynomial(self, ab, coefft, flat, derivs):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Invert the polynomial for a pair (a,b) to return (p,q).
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------
        # Initial guess
        #------------------------------------------
        pq = flat(ab, derivs=derivs)

        #------------------------------------------
        # Iterate until convergence
        #------------------------------------------
        epsilon = 1.e-14
        for iter in range(self.iters):
            r = pq.norm_sq().vals
            (cor,_) = self._compute_polynomial(r, coefft, derivs=derivs)
            pq = flat(ab, derivs=derivs)/cor
            if np.max(np.abs(1-cor)) <= epsilon: break


        #------------------------------------------
        # Fill in derivatives if necessary
        #------------------------------------------
#        if ab.derivs:
#            new_derivs = {}
#            for (key, ab_deriv) in ab.derivs.items():
#                new_derivs[key] = dpq_dab.chain(ab_deriv)

#            pq.insert_derivs(new_derivs)

## deal w derivatives

        return pq
    #=========================================================================



    #=========================================================================
    # _inverse_polynomial
    #=========================================================================
    def _inverse_polynomial(self, ab, coefft, flat, derivs):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Solve the polynomial for a pair (a,b) pair to return (p,q).
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        order = coefft.shape[0]-1

        ab = Pair.as_pair(ab, derivs)
        ab_wod = ab.wod

        #------------------------------------------
        # Make a rough initial guess
        #------------------------------------------
        pq = flat(ab, derivs=derivs)
        pq.insert_deriv('pq', Pair.IDENTITY)
###        pq.insert_deriv('uv', Pair.IDENTITY)

        #------------------------------------------
        # Iterate until convergence...
        #------------------------------------------
        prev_dpq_max = 1.e99
        for iter in range(self.iters):

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Evaluate the transform and its partial derivatives
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ab = self._forward_polynomial(pq, coefft, flat, derivs=True)
            dab_dpq = ab.d_dpq

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Apply one step of Newton's method in 2-D
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            dab = ab_wod - ab.wod

            dpq_dab = dab_dpq.reciprocal()
            dpq = dpq_dab.chain(dab)
            pq += dpq

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Test for convergence
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            dpq_max = abs(dpq).max()
            if Radial.DEBUG:
                print(iter, dpq_max)

            if dpq_max >= prev_dpq_max: break

            prev_dpq_max = dpq_max

        pq = pq.wod

        #------------------------------------------
        # Fill in derivatives if necessary
        #------------------------------------------
        if ab.derivs:
            new_derivs = {}
            for (key, ab_deriv) in ab.derivs.items():
                new_derivs[key] = dpq_dab.chain(ab_deriv)

            pq.insert_derivs(new_derivs)

        return pq
    #=========================================================================


#******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Radial(unittest.TestCase):

    def runTest(self):

        Radial.DEBUG = True

        ############################
        order = 2
        uv_from_xy_coefft = np.array([1.000, 
                                     0, 
                                    -5.9624209455667325e-08, 
                                     0, 
                                     2.7381910042256151e-14])

        uv = Pair.combos(np.arange(100), np.arange(8)) * 16
	scale = 0.00067540618
	
        fov = Radial(scale, uv.shape, coefft_uv_from_xy=uv_from_xy_coefft, uv_los=(800,64), uv_area=1.)

        xy = fov.xy_from_uv(uv)
        uv_test = fov.uv_from_xy(xy)
#        print(abs(uv - uv_test))
#        print(abs(uv - uv_test).max())
        embed()

        self.assertTrue(abs(uv - uv_test).max() < 1.e-14)





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
