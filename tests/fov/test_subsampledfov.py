################################################################################
# tests/fov/test_subsampledfov.py
################################################################################

import unittest

from oops.config import AREA_FACTOR
from oops.fov    import FlatFOV, SubsampledFOV


class Test_SubsampledFOV(unittest.TestCase):

    def runTest(self):

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

        try:
            AREA_FACTOR.old = True

            self.assertEqual(test.uv_area, 4*flat.uv_area)

            self.assertEqual(flat.area_factor((32,32)), 1.)
            self.assertEqual(test.area_factor((16,16)), 1.)

        finally:
            AREA_FACTOR.old = False

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
