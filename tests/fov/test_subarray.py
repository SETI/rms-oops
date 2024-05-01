################################################################################
# tests/fov/test_subarray.py
################################################################################

import numpy as np
import unittest

from polymath    import Pair
from oops.fov    import FlatFOV, Subarray
from oops.config import AREA_FACTOR


class Test_Subarray(unittest.TestCase):

    def runTest(self):

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
