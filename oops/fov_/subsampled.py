################################################################################
# oops/fov_/subsampled.py: Subsampled subclass of FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class Subsampled(FOV):

    PACKRAT_ARGS = ['fov', 'rescale']

    def __init__(self, fov, rescale):
        """Constructor for a Subsampled FOV.
        
        Returns a new FOV object in which the pixel size has been modified.
        The origin and the optic axis are unchanged.

        Inputs:
            fov         the FOV object within which this subsampled FOV is
                        defined.

            rescale     a single value, tuple or Pair defining the sizes of the
                        new pixels relative to the sizes of the originals.
        """

        self.fov = fov
        self.rescale  = Pair.as_pair(rescale)

        self.rescale2 = self.rescale.vals[0] * self.rescale.vals[1]
        self.rescale_mat = Matrix([[self.rescale.vals[0], 0.],
                                   [0., self.rescale.vals[1]]])
        self.rescale_inv = Matrix([[1./self.rescale.vals[0], 0.],
                                   [0., 1./self.rescale.vals[1]]])

        # Required fields
        self.uv_scale = self.fov.uv_scale.element_mul(self.rescale)
        self.uv_los   = self.fov.uv_los.element_div(self.rescale)
        self.uv_area  = self.fov.uv_area  * self.rescale2

        self.uv_shape = (self.fov.uv_shape.element_div(self.rescale)).as_int()

        assert self.rescale.element_mul(self.uv_shape) == self.fov.uv_shape

    def xy_from_uv(self, uv_pair, derivs=False, **keywords):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        return self.fov.xy_from_uv(self.rescale.element_mul(uv_pair),
                                   derivs=derivs, **keywords)

    def uv_from_xy(self, xy_pair, derivs=False, **keywords):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        uv_pair = self.fov.uv_from_xy(xy_pair, derivs=derivs, **keywords)
        uv_new = uv_pair.element_div(self.rescale)

        return uv_new

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Subsampled(unittest.TestCase):

    def runTest(self):

        # Imports just required for unit testing
        from oops.fov_.flatfov import FlatFOV

        # Centered sub-sampling...

        flat = FlatFOV((1/2048.,-1/2048.), 64)
        test = Subsampled(flat, 2)

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
        test = Subsampled(flat, 2)

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
