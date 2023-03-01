################################################################################
# oops/calibration/extendedsource.py: ExtendedSource subclass of Calibration
################################################################################

from polymath import Scalar
from oops.calibration import Calibration

class ExtendedSource(Calibration):
    """A Calibration subclass in which every pixel is multiplied by a constant
    scale factor. DEPRECATED. Use FlatCalib.

    Within a possibly distorted field of view, this is the proper calibration to
    use for extended sources.
    """

    ############################################################################
    # Methods to be defined for each Calibration subclass
    ############################################################################

    def __init__(self, name, factor):
        """Constructor for a Scaling.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      a scale scale factor to be applied to every pixel in the
                        field of view.
        """

        self.name = name
        self.factor = Scalar.as_scalar(factor)

    #### __getstate__ and __setstate__ not needed; default behavior is fine.

    #===========================================================================
    def value_from_dn(self, dn, uv_pair=None):
        """Calibrated values of an image based an uncalibrated values ("DN") and
        image coordinates.

        Input:
            dn          a scalar, numpy array or arbitrary oops Array subclass
                        containing uncalibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         an object of the same class and shape as dn, but
                        containing the calibrated values.
        """

        return dn * self.factor

    #===========================================================================
    def dn_from_value(self, value, uv_pair=None):
        """Uncalibrated image values ("DN") based on calibrated values and
        image coordinates.

        Input:
            value       a scalar, numpy array or arbitrary oops Array subclass
                        containing calibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

        return value / self.factor

    #===========================================================================
    def extended_from_dn(self, dn, uv_pair):
        return self.value_from_dn(dn, uv_pair)

    def dn_from_extended(self, value, uv_pair):
        return self.dn_from_value(value, uv_pair)

    def point_from_dn(self, dn, uv_pair):
        return self.value_from_dn(dn, uv_pair)

    def dn_from_point(self, value, uv_pair):
        return self.dn_from_value(value, uv_pair)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_ExtendedSource(unittest.TestCase):

    def runTest(self):

        import numpy as np

        es = ExtendedSource("TEST", 5.)
        self.assertEqual(es.value_from_dn(0.), 0.)
        self.assertEqual(es.value_from_dn(0., (10,10)), 0.)
        self.assertEqual(es.value_from_dn(5.), 25.)
        self.assertEqual(es.value_from_dn(5., (10,10)), 25.)
        self.assertEqual(es.value_from_dn(.5), 2.5)
        self.assertEqual(es.value_from_dn(.5, (10,10)), 2.5)

        self.assertEqual(es.dn_from_value(0.), 0.)
        self.assertEqual(es.dn_from_value(0., (10,10)), 0.)
        self.assertEqual(es.dn_from_value(25.), 5.)
        self.assertEqual(es.dn_from_value(25., (10,10)), 5.)
        self.assertEqual(es.dn_from_value(2.5), .5)
        self.assertEqual(es.dn_from_value(2.5, (10,10)), .5)

        a = Scalar(np.arange(10000).reshape((100,100)))
        self.assertEqual(a, es.dn_from_value(es.value_from_dn(a)))

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
