################################################################################
# oops/calib_/extended.py: ExtendedSource subclass of class Calibration
#
# 2/11/12 Modified (MRS) - revised for style.
# 3/20/12 MRS - New and better class name ExtendedSource.
################################################################################

from oops.calib_.calibration import Calibration
from oops.array_ import *

class ExtendedSource(Calibration):
    """A Scaling is a Calibration object in which every pixel is multiplied by a
    constant scale factor. Within a possibly distorted field of view, this is
    the proper calibration to use for extended sources.
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
        self.factor = float(factor)

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

        return dn * self.factor

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

        return dn / self.factor

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_ExtendedSource(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
