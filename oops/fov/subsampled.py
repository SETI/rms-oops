import numpy as np
import unittest

import oops
from oops.fov.FlatFOV import FlatFOV

################################################################################
# class SubsampledFOV
################################################################################

class SubsampledFOV(oops.FOV):

    def __init__(self, fov, rescale):
        """Returns a new FOV object in which the pixel size has been modified.
        The origin and the optic axis are unchanged.

        Inputs:
            fov         the FOV object within which this subarray is defined.

            rescale     a single value, tuple or Pair defining the sizes of the
                        new pixels relative to the sizes of the originals.
        """

        self.fov = fov
        self.rescale  = oops.Pair.as_pair(rescale, duplicate=True)
        self.rescale2 = self.rescale.vals[0] * self.rescale.vals[1]

        # Required fields
        self.uv_scale = self.fov.uv_scale / self.rescale
        self.uv_los   = self.fov.uv_los   / self.rescale
        self.uv_area  = self.fov.uv_area  * self.rescale2

        self.uv_shape = (self.fov.uv_shape / self.rescale).int()

    def xy_from_uv(self, uv_pair):
        """Returns a Pair of (x,y) spatial coordinates given a Pair of (u,v)
        coordinates."""

        return self.fov.xy_from_uv(self.rescale * uv_pair)

    def uv_from_xy(self, xy_pair):
        """Returns a Pair of ICS (u,v) coordinates given a Pair of (x,y) apatial
        coordinates."""

        return self.fov.uv_from_xy(xy_pair) / self.rescale

    def xy_and_dxy_duv_from_uv(self, uv_pair):
        """Returns a tuple ((x,y), dxy_duv), where the latter is the set of
        partial derivatives of (x,y) with respect to (u,v). These are returned
        as a Pair object of shape [...,2]:
            dxy_duv[...,0] = Pair((dx/du, dx/dv))
            dxy_duv[...,1] = Pair((dy/du, dy/dv))
        """

        tuple = self.fov.xy_and_dxy_duv_from_uv(self.rescale * uv_pair)
        return (tuple[0], tuple[1] * self.rescale)

########################################
# UNIT TESTS
########################################

class Test_SubsampledFOV(unittest.TestCase):

    def runTest(self):

        buffer = np.empty((51,51,2))
        buffer[:,:,0] = np.arange(0,51).reshape(51,1)
        buffer[:,:,1] = np.arange(0,51)
        uv = oops.Pair(buffer)

        flat = FlatFOV((1/2048.,-1/2048.), (0,25))
        test = SubsampledFOV(flat, 2)

        xy = test.xy_from_uv(buffer)
        self.assertEqual(xy[ 0, 0], flat.xy_from_uv((  0,  0)))
        self.assertEqual(xy[50, 0], flat.xy_from_uv((100,  0)))
        self.assertEqual(xy[ 0,50], flat.xy_from_uv((  0,100)))
        self.assertEqual(xy[50,50], flat.xy_from_uv((100,100)))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, uv)

        self.assertEqual(test.area_factor(uv), 1.)

        xy = test.xy_from_uv(uv)
        self.assertEqual(xy, flat.xy_from_uv(uv*2.))

        self.assertEqual(test.uv_area, 4*flat.uv_area)

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
