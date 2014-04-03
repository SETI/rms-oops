################################################################################
# oops/fov_/flatfov.py: Flat subclass of class FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class FlatFOV(FOV):
    """Flat is a subclass of FOV that describes a field of view that is free of
    distortion, implementing an exact pinhole camera model.
    """

    def __init__(self, uv_scale, uv_shape, uv_los=None, uv_area=None):
        """Constructor for a FlatFOV.

        The U-axis is assumed to align with X and the V-axis aligns with Y.

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

            uv_area     an optional parameter defining the nominal field of view
                        of a pixel. If not provided, the area is calculated
                        based on the area of the central pixel.
        """

        self.uv_scale = Pair.as_pair(uv_scale).as_float().as_readonly()
        if np.shape(uv_shape) == ():
            self.uv_shape = Pair((uv_shape,uv_shape)).as_readonly()
        else:
            self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float().as_readonly()

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        scale = Pair.as_pair(uv_scale).as_readonly()

        self.dxy_duv = Pair([[  scale.vals[0], 0.],
                             [0.,   scale.vals[1]]], drank=1).as_readonly()
        self.duv_dxy = Pair([[1/scale.vals[0], 0.],
                             [0., 1/scale.vals[1]]], drank=1).as_readonly()

    def uv_from_xy(self, xy_pair, derivs=False):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.
        """

        xy_pair = Pair.as_pair(xy_pair, derivs)
        return xy_pair.element_div(self.uv_scale) + self.uv_los

    def xy_from_uv(self, uv_pair, derivs=False):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.
        """

        uv_pair = Pair.as_pair(uv_pair, derivs)
        return (uv_pair - self.uv_los).element_mul(self.uv_scale)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Flat(unittest.TestCase):

    def runTest(self):

        test = FlatFOV((1/2048.,-1/2048.), 101, (50,75))

        buffer = np.empty((101,101,2))
        buffer[:,:,0] = np.arange(101).reshape(101,1)
        buffer[:,:,1] = np.arange(101)

        xy = test.xy_from_uv(buffer)
        (x,y) = xy.to_scalars()

        self.assertEqual(xy[  0,  0], (-50./2048., 75./2048.))
        self.assertEqual(xy[100,  0], ( 50./2048., 75./2048.))
        self.assertEqual(xy[  0,100], (-50./2048.,-25./2048.))
        self.assertEqual(xy[100,100], ( 50./2048.,-25./2048.))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, Pair(buffer))

        self.assertEqual(test.area_factor(buffer), 1.)

        test2 = FlatFOV((1/2048.,-1/2048.), 101, (50,75), uv_area = test.uv_area*2)
        self.assertEqual(test2.area_factor(buffer), 0.5)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
