################################################################################
# oops/fov/slicefov.py: SliceFOV subclass of FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class SliceFOV(FOV):

    def __init__(self, fov, origin, shape):
        """Returns a new FOV object in which the geometry is unchanged relative
        to another FOV, but only a slice of that FOV's (u,v) array is used. This
        differs from a Subarray FOV in that the optic axis is not modified.

        Inputs:
            fov         the reference FOV object within which this slice is
                        defined.

            origin      a tuple or Pair defining the location of the subarray's
                        pixel (0,0) in the coordinates of the reference FOV.

            shape       a single value, tuple or Pair defining the new shape of
                        the field of view in pixels.
        """

        self.fov = fov
        self.uv_origin = Pair.as_int(origin)
        self.uv_shape  = Pair.as_int(shape)

        # Required fields
        self.uv_los   = self.fov.uv_los - self.uv_origin
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v).
        """

        return self.fov.xy_from_uv(uv_pair + self.uv_origin, derivs=derivs)

    def uv_from_xy(self, xy_pair, extras=(), derivs=False):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y).
        """

        return self.fov.uv_from_xy(xy_pair, extras=extras,
                                   derivs=derivs) - self.origin

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_SliceFOV(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
