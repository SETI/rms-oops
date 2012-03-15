################################################################################
# oops_/meshgrid.py: Class Meshgrid
#
# 3/13/12 MRS - Created
################################################################################

import numpy as np

from oops_.array.all import *
from oops_.fov.fov_ import FOV

class Meshgrid(object):
    """A Meshgrid object defines a arbitrary array of coordinate pairs within
    a Field of View. It caches information about the line of sight and various
    derivatives, preventing the need for repeated calls to the FOV functions
    when the same field of view describes multiple images.

    After you create a Meshgrid object, the following are available as
    properties:

    uv              the (u,v) pairs with no derivatives
    uv_w_derivs     the (u,v) pairs with d_dlos.
    duv_dlos        the partial derivatives d(u,v)/dlos

    los             the line-of-sight unit vectors with no derivatives
    los_w_derivs    the line-of-sight unit vectors with d_duv
    dlos_duv        the partial derivatives dlos/d(u,v)
    """

    def __init__(self, fov, uv_pair, extras=()):
        """The Meshgrid constructor.

        Input:
            fov         a FOV object.
            uv_pair     a Pair object of arbitrary shape, representing (u,v)
                        coordinates into a field of view.
            extras      additional parameters that might affect the FOV
                        transform.
        """

        self.fov = fov
        self.uv = Pair.as_pair(uv_pair)
        self.extras = extras
        self.shape = self.uv.shape

        self.filled_los_w_derivs = None
        self.filled_los = None
        self.filled_uv_w_derivs = None

    @staticmethod
    def for_fov(fov, origin=0.5, undersample=1, oversample=1, limit=None,
                     swap=False, extras=()):
        """Returns a 2-D rectangular Meshgrid object for a specified sampling of
        the FOV.

        Input:
            origin      A single value, tuple or Pair defining the origin of the
                        grid. Default is 0.5, which places the first sample in
                        the middle of the first pixel.

            limit       A single value, tuple or Pair defining the upper limits
                        of the meshgrid. By default, this is the shape of the
                        FOV.

            undersample A single value, tuple or Pair defining the magnitude of
                        under-sampling to be performed. For example, a value of
                        2 would cause the meshgrid to sample every other pixel
                        along each axis.

            oversample  A single value, tuple or Pair defining the magnitude of
                        over-sampling to be performed. For example, a value of
                        2 would create a 2x2 array of samples inside each pixel.

            swap        True to swap the order of the indices in the meshgrid,
                        (v,u) instead of (u,v).

            extras      additional parameters that might affect the FOV
                        transform.
        """

        if limit is None: limit = fov.uv_shape
        limit = Pair.as_pair(limit).vals

        origin = Pair.as_pair(origin).vals
        undersample = Pair.as_pair(undersample).vals
        oversample  = Pair.as_float_pair(oversample).vals

        step = undersample/oversample
        limit += step * 1.e-12  # Allow a little slop at the upper end

        grid = Pair.meshgrid(np.arange(origin[0], limit[0], step[0]),
                             np.arange(origin[1], limit[1], step[1]))

        if swap: grid = grid.swapaxes(0,1)

        return Meshgrid(fov, grid, extras)

    @property
    def los_w_derivs(self):
        if self.filled_los_w_derivs == None:
            los = self.fov.los_from_uv(self.uv, self.extras, derivs=True)
            self.filled_los_w_derivs = los

        return self.filled_los_w_derivs

    @property
    def los(self):
        if self.filled_los == None:
            self.filled_los = self.los_w_derivs.plain()
        return self.filled_los

    @property
    def dlos_duv(self):
        return self.filled_los_w_derivs.d_duv

    @property
    def uv_w_derivs(self):
        if self.filled_uv_w_derivs == None:
            uv = self.fov.uv_from_los(self.los, self.extras, derivs=True)
            self.filled_uv_w_derivs = self.uv
            self.filled_uv_w_derivs.insert_subfield("d_dlos", uv.d_dlos)
            self.uv = self.filled_uv_w_derivs.plain()
        return self.filled_uv_w_derivs

    @property
    def duv_dlos(self):
        return self.filled_uv_w_derivs.d_dlos

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Meshgrid(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
