################################################################################
# oops/calib_/extended.py: ExtendedSource subclass of class Calibration
################################################################################

from polymath import *
from oops.calib_.calibration import Calibration

#*******************************************************************************
# ExtendedSource
#*******************************************************************************
class ExtendedSource(Calibration):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Scaling is a Calibration object in which every pixel is multiplied by a
    constant scale factor. Within a possibly distorted field of view, this is
    the proper calibration to use for extended sources.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

########################################################
# Methods to be defined for each Calibration subclass
########################################################

    PACKRAT_ARGS = ['name', 'factor']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, name, factor):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Constructor for a Scaling.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      a scale scale factor to be applied to every pixel in the
                        field of view.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.name = name
        self.factor = Scalar.as_scalar(factor)
    #===========================================================================



    #===========================================================================
    # value_from_dn
    #===========================================================================
    def value_from_dn(self, dn, uv_pair=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Returns calibrated values based an uncalibrated image value ("DN")

        and image coordinates.

        Input:
            dn          a scalar, numpy array or arbitrary oops Array subclass
                        containing uncalibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         an object of the same class and shape as dn, but
                        containing the calibrated values.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return dn * self.factor
    #===========================================================================



    #===========================================================================
    # dn_from_value
    #===========================================================================
    def dn_from_value(self, value, uv_pair=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Returns uncalibrated image values ("dn") based on calibrated values
        and image coordinates.

        Input:
            value       a scalar, numpy array or arbitrary oops Array subclass
                        containing calibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.
 
        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return value / self.factor
    #===========================================================================


#*******************************************************************************


################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_ExtendedSource
#*******************************************************************************
class Test_ExtendedSource(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
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
    #===========================================================================


#*******************************************************************************


#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
