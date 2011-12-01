import numpy as np
import unittest

import oops
from oops.fov.FlatFOV import FlatFOV

################################################################################
# class SubarrayFOV
################################################################################

class SubarrayFOV(oops.FOV):

    def __init__(self, fov, new_los, uv_shape, uv_los=None):
        """Returns a new FOV object in which the ICS origin and/or the optic
        axis have been modified.

        Inputs:
            fov         the FOV object within which this subarray is defined.

            new_los     a tuple defining the location of the subarray's line of
                        sight in the (u,v) coordinates of the original FOV.

            uv_shape    a single value, tuple or Pair defining the new size of
                        the field of view in pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the new line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.
        """

        self.fov = fov
        self.new_los_in_old_uv  = oops.Pair.as_float_pair(new_los)
        self.new_los_wrt_old_xy = fov.xy_from_uv(self.new_los_in_old_uv)

        self.uv_shape = oops.Pair.as_pair(uv_shape, duplicate=True)

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = oops.Pair.as_float_pair(uv_los, duplicate=True)

        self.new_origin_in_old_uv = self.new_los_in_old_uv - self.uv_los

        # Required fields
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def xy_from_uv(self, uv_pair):
        """Returns a Pair of (x,y) spatial coordinates given a Pair of ICS (u,v)
        coordinates."""

        return (self.fov.xy_from_uv(self.new_origin_in_old_uv + uv_pair)
                - self.new_los_wrt_old_xy)

    def uv_from_xy(self, xy_pair):
        """Returns a Pair of ICS (u,v) coordinates given a Pair of (x,y) apatial
        coordinates."""

        return (self.fov.uv_from_xy(self.new_los_wrt_old_xy + xy_pair)
                - self.new_origin_in_old_uv)

    def xy_and_dxy_duv_from_uv(self, uv_pair):
        """Returns a tuple ((x,y), dxy_duv), where the latter is the set of
        partial derivatives of (x,y) with respect to (u,v). These are returned
        as a Pair object of shape [...,2]:
            dxy_duv[...,0] = Pair((dx/du, dx/dv))
            dxy_duv[...,1] = Pair((dy/du, dy/dv))
        """

        tuple = self.fov.xy_and_dxy_duv_from_uv(self.new_origin_in_old_uv +
                                                uv_pair)
        return (tuple[0] - self.new_los_wrt_old_xy, tuple[1])

########################################
# UNIT TESTS
########################################

class Test_SubarrayFOV(unittest.TestCase):

    def runTest(self):

        flat = FlatFOV((1/2048.,-1/2048.), 101, (50,75))

        test = SubarrayFOV(flat, (50,75), 101, (50,75))
        buffer = np.empty((101,101,2))
        buffer[:,:,0] = np.arange(101).reshape(101,1)
        buffer[:,:,1] = np.arange(101)
        uv = oops.Pair(buffer)

        xy = test.xy_from_uv(buffer)
        self.assertEqual(xy, flat.xy_from_uv(uv))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, uv)

        self.assertEqual(test.area_factor(uv), 1.)

        ############################

        test = SubarrayFOV(flat, (50,75), 51)
        buffer = np.empty((51,51,2))
        buffer[:,:,0] = np.arange(51).reshape(51,1) + 0.5
        buffer[:,:,1] = np.arange(51) + 0.5
        uv = oops.Pair(buffer)

        xy = test.xy_from_uv(buffer)
        self.assertEqual(xy, -test.xy_from_uv(buffer[-1::-1,-1::-1]))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, uv)

        self.assertEqual(test.area_factor(uv), 1.)

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
