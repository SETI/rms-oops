################################################################################
# oops/fov/subsampledfov.py: SubsampledFOV subclass of FOV
################################################################################

from polymath import Pair

from . import FOV

class SubsampledFOV(FOV):
    """Subclass of FOV in which the pixels of a given base FOV class are
    re-scaled.
    """

    #===========================================================================
    def __init__(self, fov, rescale):
        """Constructor for a SubsampledFOV.

        Returns a new FOV object in which the pixel size has been modified.
        The origin and the optic axis are unchanged.

        Inputs:
            fov         the FOV object within which this subsampledFOV is
                        defined.

            rescale     a single value, tuple or Pair defining the sizes of the
                        new pixels relative to the sizes of the originals.
        """

        self.fov = fov
        self.rescale  = Pair.as_pair(rescale)
        self.rescale2 = self.rescale.vals[0] * self.rescale.vals[1]

        # Required fields
        self.uv_scale = self.fov.uv_scale.element_mul(self.rescale)
        self.uv_los   = self.fov.uv_los.element_div(self.rescale)
        self.uv_area  = self.fov.uv_area  * self.rescale2

        self.uv_shape = (self.fov.uv_shape.element_div(self.rescale)).as_int()

        assert self.rescale.element_mul(self.uv_shape) == self.fov.uv_shape

    def __getstate__(self):
        return (self.fov, self.rescale)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv_pair, tfrac=0.5, time=None, derivs=False,
                          **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        return self.fov.xy_from_uvt(self.rescale.element_mul(uv_pair),
                                    tfrac, time, derivs=derivs, **keywords)

    #===========================================================================
    def uv_from_xyt(self, xy_pair, tfrac=0.5, time=None, derivs=False,
                          **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        uv_pair = self.fov.uv_from_xyt(xy_pair, tfrac, time,
                                       derivs=derivs, **keywords)
        uv_new = uv_pair.element_div(self.rescale)

        return uv_new

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_SubsampledFOV(unittest.TestCase):

    def runTest(self):

        # Imports just required for unit testing
        from .flatfov import FlatFOV

        # Centered sub-sampling...

        flat = FlatFOV((1/2048.,-1/2048.), 64)
        test = SubsampledFOV(flat, 2)

        self.assertEqual(flat.xy_from_uv(( 0, 0)), test.xy_from_uv(( 0, 0)))
        self.assertEqual(flat.xy_from_uv(( 0,64)), test.xy_from_uv(( 0,32)))
        self.assertEqual(flat.xy_from_uv((64, 0)), test.xy_from_uv((32, 0)))
        self.assertEqual(flat.xy_from_uv((64,64)), test.xy_from_uv((32,32)))

        xy = (-32/2048., 32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        xy = (-32/2048.,-32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        xy = ( 32/2048.,-32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        xy = ( 32/2048., 32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        self.assertEqual(test.uv_area, 4*flat.uv_area)

        self.assertEqual(flat.area_factor((32,32)), 1.)
        self.assertEqual(test.area_factor((16,16)), 1.)

        # Off-center sub-sampling...

        flat = FlatFOV((1/2048.,-1/2048.), 64, uv_los=(0,32))
        test = SubsampledFOV(flat, 2)

        self.assertEqual(flat.xy_from_uv(( 0, 0)), test.xy_from_uv(( 0, 0)))
        self.assertEqual(flat.xy_from_uv(( 0,64)), test.xy_from_uv(( 0,32)))
        self.assertEqual(flat.xy_from_uv((64, 0)), test.xy_from_uv((32, 0)))
        self.assertEqual(flat.xy_from_uv((64,64)), test.xy_from_uv((32,32)))

        xy = ( 0/2048., 32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        xy = ( 0/2048.,-32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        xy = (64/2048.,-32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        xy = (64/2048., 32/2048.)
        self.assertEqual(flat.uv_from_xy(xy), test.uv_from_xy(xy) * 2.)

        self.assertEqual(test.uv_area, 4*flat.uv_area)

        self.assertEqual(flat.area_factor((32,32)), 1.)
        self.assertEqual(test.area_factor((16,16)), 1.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
