################################################################################
# oops/calib/baseclass.py: Abstract class Calibration
#
# 2/11/12 Modified (MRS) - revised for style
################################################################################

class Calibration(object):
    """Calibration is an abstract class that defines a relationship between the
    numeric values in in image array and physical quantities.
    """

########################################################
# Methods to be defined for each Calibration subclass
########################################################

    def __init__(self):
        """A constructor.

        At minimum, every Calibration object has these attributes:

            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      an approximate linear scaling factor, applicable at the
                        center of the field of view, to convert from array DN
                        values to the calibrated quantity.
        """

        pass

    def value_from_dn(self, dn, uv_pair=None):
        """Returns a Scalar value at a pixel given its image array value (DN)
        and its optional coordinate location within the FOV."""

        pass

    def dn_from_value(self, value, uv_pair=None):
        """Returns a Scalar DN value for the image array given a calibrated
        value and its optional coordinate location within the FOV."""

        pass

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Calibration(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
