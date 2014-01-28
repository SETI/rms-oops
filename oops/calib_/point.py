################################################################################
# oops_/calib/point.py: Subclass PointSource of class Calibration
#
# 2/8/12 Modified (MRS) - Changed name from AreaScaling; revised for new class
#   heirarchy.
# 3/20/12 MRS - New and better name is PointSource.
################################################################################

from oops_.calib.calibration_ import Calibration
from oops_.array.all import *

class PointSource(Calibration):
    """PointSource is a Calibration subclass in which every pixel is multiplied
    by a constant scale factor, but is also scaled by the distorted area of each
    pixel in the field of view. This compensates for the fact that larger pixels
    collect more photons. It is the appropriate calibration to use for point
    sources.
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
        self.factor = Scalar(factor)
        self.fov = fov

    def value_from_dn(self, dn, uv_pair):
        """Returns calibrated values based an uncalibrated image value ("DN")
        and image coordinates.

        Input:
            dn          a scalar, numpy array or arbitrary oops Array subclass
                        containing uncalibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         an object of the same class and shape as dn, but
                        containing the calibrated values.
        """

        value = (self.factor / self.fov.area_factor(uv_pair)) * dn

        if isinstance(dn, Array): return value
        return value.vals

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

        dn = (self.fov.area_factor(uv_pair) / self.factor) * value

        if isinstance(value, Array): return dn
        return dn.vals

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_PointSource(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
