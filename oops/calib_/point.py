################################################################################
# oops/calib_/point.py: Subclass PointSource of class Calibration
################################################################################

from polymath import *

from oops.calib_.calibration import Calibration

#*******************************************************************************
# PointSource
#*******************************************************************************
class PointSource(Calibration):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    PointSource is a Calibration subclass in which every pixel is multiplied
    by a constant scale factor, but is also scaled by the distorted area of each
    pixel in the field of view. This compensates for the fact that larger pixels
    collect more photons. It is the appropriate calibration to use for point
    sources.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ########################################################
    # Methods to be defined for each Calibration subclass
    ########################################################

    PACKRAT_ARGS = ['name', 'factor', 'fov']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, name, factor, fov):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for an Distorted Calibration.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      a scale scale factor to be applied to every pixel in the
                        field of view.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.name = name
        self.factor = Scalar.as_scalar(factor)
        self.fov = fov
    #===========================================================================



    #===========================================================================
    # value_from_dn
    #===========================================================================
    def value_from_dn(self, dn, uv_pair):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns calibrated values based an uncalibrated image value ("DN")
        and image coordinates.

        Input:
            dn          a scalar, numpy array or arbitrary oops Qube subclass
                        containing uncalibrated values.
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         an object of the same class and shape as dn, but
                        containing the calibrated values.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return dn * (self.factor / self.fov.area_factor(uv_pair))
    #===========================================================================



    #===========================================================================
    # dn_from_value
    #===========================================================================
    def dn_from_value(self, value, uv_pair):
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
        return value * (self.fov.area_factor(uv_pair) / self.factor)
    #===========================================================================



################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_PointSource(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        import numpy as np
        from oops.fov_.flatfov import FlatFOV
        from oops.constants import RPD

        flat_fov = FlatFOV((RPD/3600.,RPD/3600.), (1024,1024)) 
        ps = PointSource("TEST", 5., flat_fov)
        self.assertEqual(ps.value_from_dn(0., (512,512)), 0.)
        self.assertEqual(ps.value_from_dn(0., (10,10)), 0.)
        self.assertEqual(ps.value_from_dn(5., (512,512)), 25.)
        self.assertEqual(ps.value_from_dn(5., (10,10)), 25.)
        self.assertEqual(ps.value_from_dn(.5, (512,512)), 2.5)
        self.assertEqual(ps.value_from_dn(.5, (10,10)), 2.5)

        self.assertEqual(ps.dn_from_value(0., (512,512)), 0.)
        self.assertEqual(ps.dn_from_value(0., (10,10)), 0.)
        self.assertEqual(ps.dn_from_value(25., (512,512)), 5.)
        self.assertEqual(ps.dn_from_value(25., (10,10)), 5.)
        self.assertEqual(ps.dn_from_value(2.5, (512,512)), .5)
        self.assertEqual(ps.dn_from_value(2.5, (10,10)), .5)

        a = Scalar(np.arange(10000).reshape((100,100)))
        self.assertEqual(a, ps.dn_from_value(ps.value_from_dn(a, (10,10)), (10,10)))
    #===========================================================================



########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
