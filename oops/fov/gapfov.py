################################################################################
# oops/fov/gapfov.py: GapFOV subclass of FOV
################################################################################

import numpy as np
import numbers

from polymath import Pair
from oops.fov import FOV

class GapFOV(FOV):
    """A subclass of FOV in which there gaps between the individual pixels."""

    #===========================================================================
    def __init__(self, fov, uv_size):
        """Constructor for a GapFOV.

        Pixels in the new FOV have the same origins as in the given FOV, but
        their (u,v) extent is reduced.

        Inputs:
            fov         the FOV object relative to which this GapFOV is defined.

            uv_size     a single value, tuple or Pair defining the sizes of the
                        new pixels relative to the sizes of the originals.
        """

        self.fov = fov

        # Allow for one or two inputs
        if isinstance(uv_size, numbers.Real):
            uv_size = (uv_size, uv_size)

        # Convert to Pair
        self.uv_size = Pair.as_pair(uv_size)
        self.uv_size_inv = Pair.as_pair((1./self.uv_size.vals[0],
                                         1./self.uv_size.vals[1]))

        self._rescale2 = self.rescale.vals[0] * self.rescale.vals[1]

        # Required fields
        self.uv_scale = self.fov.uv_scale.element_mul(self.uv_size)
        self.uv_los   = self.fov.uv_los.element_div(self.rescale)
        self.uv_area  = self.fov.uv_area * self._rescale2
        self.uv_shape = self.fov.uv_shape

    def __getstate__(self):
        return (self.fov, self.uv_size)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv_pair, time=None, derivs=False, remask=False,
                                                            **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute time in seconds.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        uv_int = uv_pair.int(top=self.uv_shape, recursive=derivs)
        uv_frac = uv_pair - uv_int
        uv = uv_int + uv_frac.element_mul(self.uv_size)

        return self.fov.xy_from_uvt(uv, time=time, derivs=derivs, remask=remask,
                                                                  **keywords)

    #===========================================================================
    def uv_from_xyt(self, xy_pair, time=None, derivs=False, remask=False,
                                                            **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute time in seconds.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (x,y) locations that fall in the gaps
                        between pixels.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        uv_pair = self.fov.uv_from_xyt(xy_pair, time=time, derivs=derivs,
                                                remask=remask, **keywords)
        uv_int = uv_pair.int(top=self.uv_shape, recursive=derivs)
        uv_frac = (uv_pair - uv_int).element_mul(self.uv_size_inv)

        # Clip (u,v) in the gaps
        for k in range(2):
            mask = (uv_frac.vals[...,k] > 1.) & (uv_int.vals[...,k]
                                                  < self.uv_shape[k])
            if np.isscalar(mask):
                uv_frac.vals[...,k] = 1.
                uv_frac.mask = remask
            else:
                uv_frac.vals[...,k][mask] = 1.
                uv_frac.mask != remask

        return uv_int + uv_frac

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_GapFOV(unittest.TestCase):

    def runTest(self):

        #### TBD
        print('GapFOV unit tests are needed!')

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
