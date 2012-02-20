################################################################################
# oops/calib/scaling.py: Subclass Scaling of class Calib
#
# 2/11/12 Modified (MRS) - revised for style.
################################################################################

from baseclass import Calibration
from oops_.array_.all import *

class Scaling(Calibration):
    """A Scaling is a Calibration object in which every pixel is multiplied by a
    constant scale factor.
    """

########################################################
# Methods to be defined for each Calibration subclass
########################################################

    def __init__(self, name, factor):
        """Constructor for a Scaling.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      a scale scale factor to be applied to every pixel in the
                        field of view.
        """

        self.name = name
        self.factor = factor

    def value_from_dn(self, dn, uv_pair=None):
        """Returns a Scalar value at a pixel given its image array value (DN).
        Any given coordinate location within the FOV is ignored."""

        return Scalar.as_scalar(dn) * self.factor

    def dn_from_value(self, value, uv_pair=None):
        """Returns a Scalar DN value for the image array given a calibrated
        value. Any given coordinate location within the FOV is ignored."""

        return Scalar.as_scalar(value) / self.factor

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Scaling(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
