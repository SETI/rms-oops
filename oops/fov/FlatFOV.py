import numpy as np
import unittest

import oops

################################################################################
# FlatFOV
################################################################################

class FlatFOV(oops.FOV):
    """A FlatFOV object describes a field of view that is free of distortion,
    implementing an exact pinhole camera model.
    """

    def __init__(self, uv_scale, uv_shape, uv_los=None):
        """Constructor for a FlatFOV. The U-axis is assumed to align with X and
        the V-axis aligns with Y.

        Input:
            uv_scale    a single value, tuple or Pair defining the ratios dx/du
	                    and dy/dv. For example, if (u,v) are in units of
	                    arcseconds, then
                            uv_scale = Pair((pi/180/3600.,pi/180/3600.))
                        Use the sign of the second element to define the
                        direction of increasing V: negative for up, positive for
                        down.

            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.
        """

        self.uv_scale = oops.Pair.as_float_pair(uv_scale, duplicate=True)
        self.uv_shape = oops.Pair.as_pair(uv_shape, duplicate=True)

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = oops.Pair.as_float_pair(uv_los, duplicate=True)

        self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])

        scale = self.uv_scale.as_scalars()
        self.dxy_duv = oops.Pair([[scale[0], 0.],
                                  [0., scale[1]]])

    def xy_from_uv(self, uv_pair):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v)."""

        return (oops.Pair.as_pair(uv_pair) - self.uv_los) * self.uv_scale

    def uv_from_xy(self, xy_pair):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians."""

        return oops.Pair.as_pair(xy_pair) / self.uv_scale + self.uv_los

    def xy_and_dxy_duv_from_uv(self, uv_pair):
        """Returns a tuple ((x,y), dxy_duv), where the latter is the set of
        partial derivatives of (x,y) with respect to (u,v). These are returned
        as a Pair object of shape [...,2]:
            dxy_duv[...,0] = Pair((dx/du, dx/dv))
            dxy_duv[...,1] = Pair((dy/du, dy/dv))
        """

        return (self.xy_from_uv(uv_pair), self.dxy_duv)

########################################
# UNIT TESTS
########################################

class Test_FlatFOV(unittest.TestCase):

    def runTest(self):

        test = FlatFOV((1/2048.,-1/2048.), 101, (50,75))

        buffer = np.empty((101,101,2))
        buffer[:,:,0] = np.arange(101).reshape(101,1)
        buffer[:,:,1] = np.arange(101)

        xy = test.xy_from_uv(buffer)
        (x,y) = xy.as_scalars()

        self.assertEqual(xy[  0,  0], (-50./2048., 75./2048.))
        self.assertEqual(xy[100,  0], ( 50./2048., 75./2048.))
        self.assertEqual(xy[  0,100], (-50./2048.,-25./2048.))
        self.assertEqual(xy[100,100], ( 50./2048.,-25./2048.))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, oops.Pair(buffer))

        self.assertEqual(test.area_factor(buffer), 1.)

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
