################################################################################
# oops_/fov/polynomial.py: Polynomial subclass of FOV
#
# 2/1/12 Modified (MRS) - copy() added to as_pair() calls.
# 2/2/12 Modified (MRS) - converted to new class names and hierarchy.
# 2/23/12 MRS - Gave each method the option to return partial derivatives.
################################################################################

import numpy as np

from oops_.fov.fov_ import FOV
from oops_.array.all import *

class Polynomial(FOV):
    """The Polynomial subclass of FOV describes a field of view in which the
    distortion is described by a 2-D polynomial. This is the approached used by
    Space Telescope Science Institute to describe the Hubble instrument fields
    of view. A Polynomial FOV has no dependence on the optional extra indices
    that can be associated with time, wavelength band, etc.
    """

    def __init__(self, uv_coefft, uv_shape, uv_los=None, uv_area=None):
        """Constructor for a PolynomialFOV.

        Inputs:
            uv_coefft   the coefficient array of the polynomial. The array has
                        shape [order+1,order+1,2], where uv_coefft[i,j,0] is the
                        coefficient on (u**i,v**j) yielding x(u,v), and
                        uv_coefft[i,j,1] is the coefficient yielding y(u,v). All
                        coefficients are 0 for (i+j) > order.

            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal area of a
                        pixel in steradians after distortion has been removed.
        """

        self.uv_coefft = np.asarray(uv_coefft)
        self.order     = self.uv_coefft.shape[0] - 1

        self.uv_shape = Pair.as_pair(uv_shape).copy()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_float(uv_los, copy=True)

        # Required attribute
        self.uv_scale = Pair.as_pair((uv_coefft[1,0,0], uv_coefft[0,1,1]))

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        # Required attribute
        self.uv_scale = Pair.as_pair((uv_coefft[1,0,0], uv_coefft[0,1,1]))

    ########################################

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        # Subtract off the center of the field of view
        uv_pair = Pair.as_pair(uv_pair) - self.uv_los
        (du,dv) = uv_pair.as_scalars()
        du = du.vals[..., np.newaxis]
        dv = dv.vals[..., np.newaxis]

        # Construct the powers of line and sample
        du_powers = [1.]
        dv_powers = [1.]
        for k in range(1, self.order + 1):
            du_powers.append(du_powers[-1] * du)
            dv_powers.append(dv_powers[-1] * dv)

        # Initialize the output
        xy_pair_vals = np.zeros(uv_pair.shape + [2])

        # Evaluate the polynomials
        #
        # Start with the high-order terms and work downward, because this
        # improves accuracy. Stop at one because there are no zero-order terms.
        for k in range(self.order, 0, -1):
          for i in range(k+1):
            j = k - i
            xy_pair_vals += self.uv_coefft[i,j,:] * du_powers[i] * dv_powers[j]

        xy = Pair(xy_pair_vals, uv_pair.mask)

        # Calculate and return derivatives if necessary
        if derivs:
            dxy_duv_vals = np.zeros(uv_pair.shape + [2,2])

            for k in range(self.order, 0, -1):
              for i in range(k+1):
                j = k - i
                dxy_duv_vals[...,:,0] += (self.uv_coefft[i,j,:] *
                                          i*du_powers[i-1] * dv_powers[j])
                dxy_duv_vals[...,:,1] += (self.uv_coefft[i,j,:] *
                                          du_powers[i] * j*dv_powers[j-1])

            xy.insert_subfield("d_duv", MatrixN(dxy_duv_vals, xy.mask))

        return xy

    ########################################

    def uv_from_xy(self, xy_pair, extras=(), derivs=False, iters=3):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y) as a MatrixN with item
        shape [2,2].
        """

        # Make a rough initial guess
        xy_pair = Pair.as_pair(xy_pair)
        uv_test = xy_pair / self.uv_scale + self.uv_los

        # Iterate a fixed number of times...
        for iter in range(iters):

            # Evaluate the transform and its partial derivatives
            xy_test = self.xy_from_uv(uv_test, derivs=True)
            dxy_duv = xy_test.d_duv

            # Apply one step of Newton's method in 2-D
            dxy = xy_pair - xy_test

            duv_dxy = ~dxy_duv
            uv_test += duv_dxy * (xy_pair - xy_test)

            # print iter, max(np.max(np.abs(du)),np.max(np.abs(dv)))

        if derivs:
            uv_test.insert_subfield("d_dxy", duv_dxy)

        return uv_test

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Polynomial(unittest.TestCase):

    def runTest(self):

        # Tested fully by Instrument.HST module

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
