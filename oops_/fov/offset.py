################################################################################
# oops_/fov/offset.py: Offset subclass of FOV
#
# 3/21/12 MRS - New.
# 10/28/12 MRS - Complete update to accommodate the Fittable interface.
################################################################################

import numpy as np

from oops_.fov.fov_ import FOV
from oops.fittable  import Fittable
from oops_.array.all import *

class Offset(FOV, Fittable):

    def __init__(self, fov, uv_offset=(0.,0.)):
        """Returns a new FOV object in which the line of sight has been shifted
        by a specified distance in units of pixels relative to another FOV. This
        is typically used for image navigation and pointing corrections.

        Inputs:
            fov         the FOV object from which this subarray has been offset.

            uv_offset   a tuple or Pair defining the offset of the new FOV
                        relative to the old. This can be understood as having
                        the effect of shifting predicted image geometry relative
                        to what the image actually shows.
        """

        self.fov = fov
        self.uv_offset = Pair.as_float(uv_offset)

        # Required attributes for FOV
        self.uv_shape = self.fov.uv_shape
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

        # Required attributes for Fittable
        self.nparams = 2
        self.cachde  = {}   # not used

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        new_xy = self.fov.xy_from_uv(uv_pair - self.uv_offset, extras, derivs)

        if derivs:
            new_xy.insert_subfield("d_uv", old_xy.d_duv)

        return new_xy

    def uv_from_xy(self, xy_pair, extras=(), derivs=False):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y) as a MatrixN with item
        shape [2,2].
        """

        new_uv = self.fov.uv_from_xy(xy_pair, extras, derivs) + self.uv_offset

        if derivs:
            new_uv.insert_subfield("d_dxy", old_uv.d_duv)

        return new_uv

    ########################################
    # Fittable interface
    ########################################

    def set_params(self, params):
        """Redefines the Fittable object, using this set of parameters. Unlike
        method set_params(), this method does not check the cache first.
        Override this method if the subclass should use a cache.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        self.uv_offset = Pair(params)

    def get_params(self):
        """Returns the current set of parameters defining this fittable object.

        Return:         a Numpy 1-D array of floating-point numbers containing
                        the parameter values defining this object.
        """

        return self.uv_offset.vals

    def copy(self):
        """Returns a deep copy of the given object. The copy can be safely
        modified without affecting the original."""

        return Offset(self.fov, self.uv_offset)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Offset(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
