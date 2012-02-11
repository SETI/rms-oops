import numpy as np
import unittest

import oops

################################################################################
# PolynomialFOV
################################################################################

class PolynomialFOV(oops.FOV):
    """A PolynomialFOV describes a field of view in which the distortion is
    described by a 2-D polynomial. This is the approached used by Space
    Telescope Science Institute to describe the Hubble instrument fields of
    view.
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

            uv_area     the nominal area of a pixel in steradians after
                        distortion has been removed.
        """

        self.uv_coefft = np.asarray(uv_coefft)
        self.order     = self.uv_coefft.shape[0] - 1

        self.uv_shape = oops.Pair.as_pair(uv_shape, duplicate=True)

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = oops.Pair.as_float_pair(uv_los, duplicate=True)

        # Required attribute
        self.uv_scale = oops.Pair.as_pair((uv_coefft[1,0,0], uv_coefft[0,1,1]))

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        # Required attribute
        self.uv_scale = oops.Pair.as_pair((uv_coefft[1,0,0], uv_coefft[0,1,1]))

    ########################################

    def xy_from_uv(self, uv_pair):
        """Returns a Pair of (x,y) spatial coordinates given a Pair of (u,v)
        coordinates."""

        # Subtract off the center of the field of view
        uv_pair = oops.Pair.as_pair(uv_pair) - self.uv_los
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

        return oops.Pair(xy_pair_vals)

    ########################################

    def xy_and_dxy_duv_from_uv(self, uv_pair):
        """Returns a tuple ((x,y), dxy_duv), where the latter is the set of
        partial derivatives of (x,y) with respect to (u,v). These are returned
        as a Pair object of shape [...,2]:
            dxy_duv[...,0] = Pair((dx/du, dx/dv))
            dxy_duv[...,1] = Pair((dy/du, dy/dv))
        """

        # Subtract off the center of the field of view
        uv_pair = oops.Pair.as_pair(uv_pair) - self.uv_los
        (du,dv)  = uv_pair.as_scalars()
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
        dxy_duv_vals = np.zeros(uv_pair.shape + [2,2])

        # Evaluate the polynomials
        #
        # Start with the high-order terms and work downward, because this
        # improves accuracy. Stop at one because there are no zero-order terms.
        for k in range(self.order, 0, -1):
          for i in range(k+1):
            j = k - i
            xy_pair_vals += self.uv_coefft[i,j,:] * du_powers[i] * dv_powers[j]

            dxy_duv_vals[...,:,0] += (self.uv_coefft[i,j,:] *
                                      i*du_powers[i-1] * dv_powers[j])
            dxy_duv_vals[...,:,1] += (self.uv_coefft[i,j,:] *
                                      du_powers[i] * j*dv_powers[j-1])

        return (oops.Pair(xy_pair_vals), oops.Pair(dxy_duv_vals))

    ########################################

    def uv_from_xy(self, xy_pair, iters=3):
        """Returns a Pair of (u,v) coordinates given a Pair of (x,y) spatial
        coordinates."""

        # Make a rough initial guess
        xy_pair = oops.Pair.as_pair(xy_pair)
        uv_test = xy_pair / self.uv_scale + self.uv_los

        # Iterate a fixed number of times...
        for iter in range(iters):

            # Evaluate the transform and its partial derivatives
            (xy_test, dxy_duv) = self.xy_and_dxy_duv_from_uv(uv_test)

            # Apply one step of Newton's method in 2-D
            dxy = xy_pair.vals - xy_test.vals
            dx = dxy[...,0]
            dy = dxy[...,1]

            dx_du = dxy_duv.vals[...,0,0]
            dx_dv = dxy_duv.vals[...,0,1]
            dy_du = dxy_duv.vals[...,1,0]
            dy_dv = dxy_duv.vals[...,1,1]

            discr = dx_du * dy_dv - dy_du * dx_dv
            du = (dx * dy_dv - dy * dx_dv) / discr
            dv = (dy * dx_du - dx * dy_du) / discr

            uv_test.vals[...,0] += du
            uv_test.vals[...,1] += dv

            # print iter, max(np.max(np.abs(du)),np.max(np.abs(dv)))

        return uv_test

########################################
# UNIT TESTS
########################################

class Test_PolynomialFOV(unittest.TestCase):

    def runTest(self):

        # Tested fully by Instrument.HST module

        pass

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
