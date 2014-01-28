################################################################################
# oops/calib_/calibration.py: Abstract class Calibration
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
        """Returns calibrated values based an uncalibrated image value ("DN")
        and image coordinates.

        Input:
            dn          a scalar, numpy array or arbitrary oops Array subclass
                        containing uncalibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         an object of the same class and shape as dn, but
                        containing the calibrated values.
        """

        pass

    def dn_from_value(self, value, uv_pair=None):
        """Returns uncalibrated image values ("dn") based on calibrated values
        and image coordinates.

        Input:
            value       a scalar, numpy array or arbitrary oops Array subclass
                        containing calibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.
 
        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

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
