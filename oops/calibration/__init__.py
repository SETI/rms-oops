################################################################################
# oops/calibration/__init__.py
################################################################################

import numpy as np
from polymath import Pair

class Calibration(object):
    """Calibration is an abstract class that defines a relationship between the
    numeric values in in image array and physical quantities.

    All subclasses have at least these two attributes:
        name        the name of the quantity that this Calibration converts to.
        factor      the value or array that multiplies DN values.
        baseline    an offset value that is subtracted from each DN before the
                    factor is applied.
        shape       the broadcasted shape of the factor and the baseline. When
                    applying the Calibration to a data object, the data object
                    (excluding spatial indices) must be broadcastable to this
                    shape.
        fov         the FOV object to which this calibration refers. Could be
                    None if the object does not require an FOV.
    """

    ############################################################################
    # Methods to be defined for each Calibration subclass
    ############################################################################

    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Input:
            dn          a Scalar or array of un-calibrated image array values at
                        the given pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image. Note
                        the dn and uv_pair will be casted to the same shape.

        Return:         calibrated values.
        """

        raise NotImplementedError(type(self).__name__ + '.extended_from_dn '
                                  'is not implemented')

    #===========================================================================
    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Input:
            value       a Scalar or array of calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image. Note
                        the dn and uv_pair will be casted to the same shape.

        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

        raise NotImplementedError(type(self).__name__ + '.dn_from_extended '
                                  'is not implemented')

    #===========================================================================
    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Input:
            dn          a Scalar or array of un-calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image. Note
                        the dn and uv_pair will be casted to the same shape.

        Return:         calibrated values.
        """

        raise NotImplementedError(type(self).__name__ + '.point_from_dn '
                                  'is not implemented')

    #===========================================================================
    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Input:
            value       a Scalar or array of calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image. Note
                        the dn and uv_pair will be casted to the same shape.

        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

        raise NotImplementedError(type(self).__name__ + '.dn_from_extended '
                                  'is not implemented')

    #===========================================================================
    def prescale(self, factor, baseline=0., name=''):
        """A version of this Calibration in which image DNs are re-scaled before
        the calibration is applied.

        Input:
            factor      scale factor to apply to DN values.
            baseline    an optional baseline value to subtract from every DN
                        value before applying the new scale factor.
            name        optional new name. If blank, the existing name is
                        preserved.

        Return:         a new object with the given scale factor and baseline
                        incorporated.
        """

        raise NotImplementedError(type(self).__name__ + '.prescale '
                                  'is not implemented')

    ############################################################################
    # Methods probably not requiring overrides
    ############################################################################

    def value_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        DEPRECATED. Use extended_from_dn or point_from_dn.

        Input:
            dn          a Scalar or array of un-calibrated image array values at
                        the given pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image. Note
                        the dn and uv_pair will be casted to the same shape.

        Return:         calibrated values.
        """

        return self.extended_from_dn(dn, uv_pair)

    #===========================================================================
    def dn_from_value(self, value, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        DEPRECATED. Use dn_from_extended or dn_from_point.

        Input:
            value       a Scalar or array of calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image. Note
                        the dn and uv_pair will be casted to the same shape.

        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

        return self.dn_from_extended(value, uv_pair)

    ############################################################################
    # Support methods
    ############################################################################

    def area_factor(self, uv_pair):
        """Relative pixel area relative to the center of the field of view.

        Requires that the class have an attribute "fov", containing either the
        FOV object or an area map.

        Input:
            uv_pair     a Pair containing (u,v) indices into the image.

        Return:         area factors.
        """

        if isinstance(self.fov, np.ndarray):
            uv_pair = Pair.as_pair(uv_pair, recursive=False)
            uv = uv_pair.int(self.fov.shape, clip=True)
            return self.fov[uv.vals[...,0], uv.vals[...,1]]

        return self.fov.area_factor(uv_pair)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Calibration(unittest.TestCase):

    def runTest(self):

        # No tests here - this is just an abstract superclass

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
