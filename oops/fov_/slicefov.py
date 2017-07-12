################################################################################
# oops/fov_/slicefov.py: SliceFOV subclass of FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class SliceFOV(FOV):
    """SliceFOV is a subclass of FOV in only a slice of another FOV's (u,v)
    array is used, but the geometry is unchanged. This differs from a Subarray
    in that the optic axis is not modified.
    """

    def __init__(self, fov, origin, shape):
        """Constructor for a SliceFOV.

        Inputs:
            fov         the reference FOV object within which this slice is
                        defined.

            origin      a tuple or Pair defining the location of the subarray's
                        pixel (0,0) in the coordinates of the reference FOV.

            shape       a single value, tuple or Pair defining the new shape of
                        the field of view in pixels.

        """

        self.fov = fov
        self.uv_origin = Pair.as_int(origin).as_readonly()
        self.uv_shape  = Pair.as_int(shape).as_readonly()

        # Required fields
        self.uv_los   = self.fov.uv_los - self.uv_origin
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def xy_from_uv(self, uv_pair, derivs=False, **keywords):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        return self.fov.xy_from_uv(uv_pair + self.uv_origin, derivs=derivs,
                                                             **keywords)

    def uv_from_xy(self, xy_pair, extras=(), derivs=False, **keywords):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        return self.fov.uv_from_xy(xy_pair, derivs=derivs, **keywords) - \
               self.origin

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
