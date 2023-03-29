################################################################################
# oops/fov/subarray.py: Subarray subclass of FOV
################################################################################

import numpy as np
from polymath import Pair
from oops.fov import FOV

class Subarray(FOV):
    """Subclass of FOV that describes a rectangular region of a larger FOV."""

    #===========================================================================
    def __init__(self, fov, new_los, uv_shape, uv_los=None):
        """Constructor for a Subarray.

        In the returned FOV object, the ICS origin and/or the optic axis have
        been modified.

        Input:
            fov         the FOV object within which this subarray is defined.

            new_los     a tuple or Pair defining the location of the subarray's
                        line of sight in the (u,v) coordinates of the original
                        FOV.

            uv_shape    a single value, tuple or Pair defining the new size of
                        the field of view in pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the new line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.
        """

        self.fov = fov
        self.new_los_in_old_uv  = Pair.as_pair(new_los).as_float()
        self.new_los_wrt_old_xy = fov.xy_from_uv(self.new_los_in_old_uv)
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_readonly()

        self.new_origin_in_old_uv = self.new_los_in_old_uv - self.uv_los

        self.new_los_in_old_uv.as_readonly()
        self.new_los_wrt_old_xy.as_readonly()
        self.uv_shape.as_readonly()
        self.uv_los.as_readonly()
        self.new_origin_in_old_uv.as_readonly

        # Required fields
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def __getstate__(self):
        return (self.fov, self.new_los, self.uv_shape, self.uv_los)

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

        old_xy = self.fov.xy_from_uvt(self.new_origin_in_old_uv + uv_pair,
                                      time=time, derivs=derivs, remask=remask,
                                      **keywords)
        return old_xy - self.new_los_wrt_old_xy

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

        old_uv = self.fov.uv_from_xyt(self.new_los_wrt_old_xy + xy_pair,
                                      time=time, derivs=derivs, remask=remask,
                                      **keywords)
        return old_uv - self.new_origin_in_old_uv

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Subarray(unittest.TestCase):

    def runTest(self):

        # Imports just required for unit testing
        from oops.fov.flatfov import FlatFOV
        from oops.config      import AREA_FACTOR

        try:
            AREA_FACTOR.old = True

            flat = FlatFOV((1/2048.,-1/2048.), 101, (50,75))

            test = Subarray(flat, (50,75), 101, (50,75))
            buffer = np.empty((101,101,2))
            buffer[:,:,0] = np.arange(101).reshape(101,1)
            buffer[:,:,1] = np.arange(101)
            uv = Pair(buffer)

            xy = test.xy_from_uv(buffer)
            self.assertEqual(xy, flat.xy_from_uv(uv))

            uv_test = test.uv_from_xy(xy)
            self.assertEqual(uv_test, uv)

            self.assertEqual(test.area_factor(uv), 1.)

            ############################

            test = Subarray(flat, (50,75), 51)
            buffer = np.empty((51,51,2))
            buffer[:,:,0] = np.arange(51).reshape(51,1) + 0.5
            buffer[:,:,1] = np.arange(51) + 0.5
            uv = Pair(buffer)

            xy = test.xy_from_uv(buffer)
            self.assertEqual(xy, -test.xy_from_uv(buffer[-1::-1,-1::-1]))

            uv_test = test.uv_from_xy(xy)
            self.assertEqual(uv_test, uv)

            self.assertEqual(test.area_factor(uv), 1.)

        finally:
            AREA_FACTOR.old = False

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
