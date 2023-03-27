################################################################################
# oops/fov/slicefov.py: SliceFOV subclass of FOV
################################################################################

from polymath import Pair
from oops.fov import FOV

class SliceFOV(FOV):
    """A subclass of FOV in which only a slice of another FOV's (u,v) array is
    used, but the geometry is unchanged.

    This differs from a Subarray in that the optic axis is not modified.
    """

    #===========================================================================
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
        self.uv_origin = Pair.as_pair(origin).as_int().as_readonly()
        self.uv_shape  = Pair.as_pair(shape).as_int().as_readonly()

        # Required fields
        self.uv_los   = self.fov.uv_los - self.uv_origin
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def __getstate__(self):
        return (self.fov, self.uv_origin, self.shape)

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

        return self.fov.xy_from_uvt(uv_pair + self.uv_origin, time=time,
                                    derivs=derivs, remask=remask, **keywords)

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
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        new_xy = self.fov.uv_from_xy(xy_pair, time=time, derivs=derivs,
                                              remask=remask, **keywords)
        return new_xy - self.origin

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
