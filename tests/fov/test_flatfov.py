################################################################################
# tests/fov/test_flatfov.py
################################################################################

import unittest
import numpy as np

from polymath import Pair
from oops.fov import FlatFOV


class Test_FlatFOV(unittest.TestCase):

    def runTest(self):

        from oops.config import AREA_FACTOR

        test = FlatFOV((1/2048.,-1/2048.), (101,101), (50,75))

        buffer = np.empty((101,101,2))
        buffer[:,:,0] = np.arange(101).reshape(101,1)
        buffer[:,:,1] = np.arange(101)

        xy = test.xy_from_uv(buffer)
        (x,y) = xy.to_scalars()

        self.assertEqual(xy[  0,  0], (-50./2048., 75./2048.))
        self.assertEqual(xy[100,  0], ( 50./2048., 75./2048.))
        self.assertEqual(xy[  0,100], (-50./2048.,-25./2048.))
        self.assertEqual(xy[100,100], ( 50./2048.,-25./2048.))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, Pair(buffer))

        try:
            AREA_FACTOR.old = True
            self.assertEqual(test.area_factor(buffer), 1.)

            test2 = FlatFOV((1/2048.,-1/2048.), 101, (50,75),
                            uv_area=test.uv_area*2)
            self.assertEqual(test2.area_factor(buffer), 0.5)

        finally:
            AREA_FACTOR.old = False

        # Test offset_angles_from_duv and offset_duv_from_angles
        fov = FlatFOV((1/2048.,-1/2048.), (101,101), (50,75))
        uvlist = ([0,0], [0,101], [101,0], [50,75], fov.uv_shape/2., fov.uv_shape)
        uvlist = [Pair.as_pair(uv) for uv in uvlist]

        for uv0 in uvlist:
            for uv1 in uvlist:
                duv = uv1 - uv0
                angles = fov.offset_angles_from_duv(duv, origin=uv0)
                test = fov.offset_duv_from_angles(angles, origin=uv0)
                self.assertLess((test - duv).norm(), 1.e-13)


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
