################################################################################
# oops/fov/subarray.py: Subarray subclass of FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class Subarray(FOV):

    def __init__(self, fov, new_los, uv_shape, uv_los=None):
        """Returns a new FOV object in which the ICS origin and/or the optic
        axis have been modified.

        Inputs:
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

        if np.shape(uv_shape) == () or len(np.asarray(uv_shape)) == 1:
            self.uv_shape = Pair.as_pair((uv_shape, uv_shape)).copy()
        else:
            self.uv_shape = Pair.as_pair(uv_shape).copy()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            if np.shape(uv_los) == () or len(np.asarray(uv_los)) == 1:
                self.uv_los = Pair.as_pair((uv_los, uv_los)).copy()
            else:
                self.uv_los = Pair.as_pair(uv_los).copy()

        self.new_origin_in_old_uv = self.new_los_in_old_uv - self.uv_los

        # Required fields
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        old_xy = self.fov.xy_from_uv(self.new_origin_in_old_uv + uv_pair,
                                     extras, derivs)
        new_xy = old_xy - self.new_los_wrt_old_xy

        if derivs:
            new_xy.insert_deriv("uv", old_xy.d_duv)

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

        old_uv = self.fov.uv_from_xy(self.new_los_wrt_old_xy + xy_pair,
                                     extras, derivs)
        new_uv = old_uv - self.new_origin_in_old_uv

        if derivs:
            new_uv.insert_deriv("xy", old_uv.d_dxy)

        return new_uv

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Subarray(unittest.TestCase):

    def runTest(self):

        # Imports just required for unit testing
        from oops.fov_.flatfov import FlatFOV

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

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
