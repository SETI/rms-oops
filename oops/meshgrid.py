################################################################################
# oops/meshgrid.py: Class Meshgrid
################################################################################

import numpy as np
import numbers
from polymath import *

from oops.fov_.fov import FOV

class Meshgrid(object):
    """A Meshgrid object defines a arbitrary array of coordinate pairs within
    a Field of View. It caches information about the line of sight and various
    derivatives, preventing the need for repeated calls to the FOV functions
    when the same field of view describes multiple images.

    After you create a Meshgrid object, the following are available as
    properties:

    uv              the (u,v) pairs with no derivatives.
    uv_w_derivs     the (u,v) pairs with d_dlos.
    duv_dlos        the partial derivatives d(u,v)/dlos.

    los             the line-of-sight unit vectors with no derivatives.
    los_w_derivs    the line-of-sight unit vectors with d_duv.
    dlos_duv        the partial derivatives dlos/d(u,v).
    """

    def __init__(self, fov, uv_pair, fov_keywords={}):
        """The Meshgrid constructor.

        Input:
            fov         a FOV object.
            uv_pair     a Pair object of arbitrary shape, representing (u,v)
                        coordinates within a field of view.
            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        self.fov = fov
        self.uv = Pair.as_pair(uv_pair).without_derivs()
        self.fov_keywords = fov_keywords
        self.shape = self.uv.shape

        self.filled_los_w_derivs = None
        self.filled_los = None
        self.filled_uv_w_derivs = None

    @staticmethod
    def for_fov(fov, origin=0.5, undersample=1, oversample=1, limit=None,
                     swap=False, fov_keywords={}):
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

            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        # Convert inputs to NumPy 2-element arrays
        if limit is None: limit = fov.uv_shape
        if isinstance(limit, numbers.Number): limit = (limit,limit)
        limit = Pair.as_pair(limit).values

        if isinstance(origin, numbers.Number): origin = (origin, origin)
        origin = Pair.as_pair(origin).values

        if isinstance(undersample, numbers.Number):
            undersample = (undersample, undersample)
        undersample = Pair.as_pair(undersample).values

        if isinstance(oversample, numbers.Number):
            oversample = (oversample, oversample)
        oversample = Pair.as_pair(oversample).values

        step = undersample/oversample
        limit = limit + step * 1.e-10  # Allow a little slop at the upper end

        urange = np.arange(origin[0], limit[0], step[0])
        vrange = np.arange(origin[1], limit[1], step[1])

        usize = urange.size
        vsize = vrange.size

        if usize == 1: urange = np.array(urange[0])
        if vsize == 1: vrange = np.array(vrange[0])

        grid = Pair.combos(urange, vrange).values
        if usize > 1 and vsize > 1 and swap: grid = grid.swapaxes(0,1)

        return Meshgrid(fov, grid, fov_keywords)

    @property
    def los_w_derivs(self):
        if self.filled_los_w_derivs is None:
            uv = self.uv.with_deriv('uv', Pair.IDENTITY, 'insert')
            los = self.fov.los_from_uv(uv, derivs=True, **self.fov_keywords)
            self.filled_los_w_derivs = los

        return self.filled_los_w_derivs

    @property
    def los(self):
        if self.filled_los is None:
            if self.filled_los_w_derivs:
                self.filled_los = self.filled_los_w_derivs.without_derivs()
            else:
                self.filled_los = self.fov.los_from_uv(self.uv, derivs=False,
                                                       **self.fov_keywords)

        return self.filled_los

    @property
    def dlos_duv(self):
        return self.los_w_derivs.d_duv

    @property
    def uv_w_derivs(self):
        if self.filled_uv_w_derivs is None:
            los = self.los.with_deriv('los', Vector3.IDENTITY, 'insert')
            uv = self.fov.uv_from_los(los, derivs=True, **self.fov_keywords)
            self.filled_uv_w_derivs = self.uv.with_deriv('los', uv.d_dlos)

        return self.filled_uv_w_derivs

    @property
    def duv_dlos(self):
        return self.uv_w_derivs.d_dlos

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
