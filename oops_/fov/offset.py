################################################################################
# oops_/fov/offset.py: Offset subclass of FOV
#
# 3/21/12 MRS - New.
################################################################################

import numpy as np

from oops_.fov.subarray import Subarray
from oops_.array.all import *

class Offset(Subarray):

    def __init__(self, fov, uv_offset):
        """Returns a new FOV object in which the line of sight has been shifted
        by a specified distance in units of pixels relative to another FOV. This
        is typically used for image navigation and pointing corrections.

        Inputs:
            fov         the FOV object from which this subarray has been offset.

            uv_offset   a tuple or Pair defining the offset of the new FOV
                        relative to the old. This can be understood as having
                        the effect of shifting predicted image geometry relative
                        to what the image actually shows.
        """

        new_center = fov.uv_from_xy((0,0)) + Pair.as_float_pair(uv_offset)

        return Subarray.__init__(self, fov, new_center, fov.uv_shape,
                                       new_center)

 ################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Offset(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
