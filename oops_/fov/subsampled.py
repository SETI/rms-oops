################################################################################
# oops_/fov/subsampled.py: Subsampled subclass of FOV
#
# 2/1/12 Modified (MRS) - copy() added to as_pair() calls.
# 2/2/12 Modified (MRS) - converted to new class names and hierarchy.
# 2/23/12 MRS - Gave each method the option to return partial derivatives.
################################################################################

import numpy as np

from oops_.fov.fov_ import FOV
from oops_.array.all import *

class Subsampled(FOV):

    def __init__(self, fov, rescale):
        """Returns a new FOV object in which the pixel size has been modified.
        The origin and the optic axis are unchanged.

        Inputs:
            fov         the FOV object within which this subsampled FOV is
                        defined.

            rescale     a single value, tuple or Pair defining the sizes of the
                        new pixels relative to the sizes of the originals.
        """

        self.fov = fov
        self.rescale  = Pair.as_pair(rescale).copy()
        self.rescale2 = self.rescale.vals[0] * self.rescale.vals[1]

        # Required fields
        self.uv_scale = self.fov.uv_scale / self.rescale
        self.uv_los   = self.fov.uv_los   / self.rescale
        self.uv_area  = self.fov.uv_area  * self.rescale2

        self.uv_shape = (self.fov.uv_shape / self.rescale).int()

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        xy = self.fov.xy_from_uv(self.rescale * uv_pair, extras, derivs)

        if derivs:
            xy.insert_subfield("d_uv", xy.d_duv * self.rescale.vals)

        return xy

    def uv_from_xy(self, xy_pair, extras=(), derivs=False):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y) as a MatrixN with item
        shape [2,2].
        """

        uv_old = self.fov.uv_from_xy(xy_pair, extras, derivs)
        uv_new = uv_old / self.rescale

        if derivs is True:
            uv_new.insert_subfield("d_dxy", uv_old.d_dxy.copy())
            uv_new.d_dxy.vals /= self.rescale.vals

        return uv_new

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Subsampled(unittest.TestCase):

    def runTest(self):

        # Imports just required for unit testing
        from flat import Flat

        buffer = np.empty((51,51,2))
        buffer[:,:,0] = np.arange(0,51).reshape(51,1)
        buffer[:,:,1] = np.arange(0,51)
        uv = Pair(buffer)

        flat = Flat((1/2048.,-1/2048.), (0,25))
        test = Subsampled(flat, 2)

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

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
