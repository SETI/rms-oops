################################################################################
# oops_/calib/distorted.py: Subclass Distorted of class Calib
#
# 2/8/12 Modified (MRS) - Changed name from AreaScaling; revised for new class
#   heirarchy.
################################################################################

from oops_.calib.calibration_ import Calibration
from oops_.array.all import *

class Distorted(Calibration):
    """Distorted is a Calibration subclass in which every pixel is multiplied by
    a constant scale factor, but is also scaled by the distorted area of each
    pixel in the field of view. This compensates for the fact that larger pixels
    collect more photons.
    """

########################################################
# Methods to be defined for each Calibration subclass
########################################################

    def __init__(self, name, factor, fov):
        """Constructor for an Distorted Calibration.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      a scale scale factor to be applied to every pixel in the
                        field of view.
        """

        self.name = name
        self.factor = factor
        self.fov = fov

    def value_from_dn(self, dn, uv_pair=None):
        """Returns a Scalar value at a pixel given its image array value (DN)
        and its coordinate location within the FOV."""

        if uv_pair is None:
            uv_pair = self.fov.uv_los
        return Scalar.as_scalar(dn) * (self.factor /
                                       self.fov.area_factor(uv_pair))

    def dn_from_value(self, value, uv_pair=None):
        """Returns a Scalar DN value for the image array given a calibrated
        value and its coordinate location within the FOV."""

        if uv_pair is None:
            uv_pair = self.fov.uv_los

        return Scalar.as_scalar(value) * (self.fov.area_factor(uv_pair) /
                                          self.factor)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Distorted(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
