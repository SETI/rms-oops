##########################################################################################
# oops/calibration/calibration_.py
##########################################################################################

import numpy as np
from polymath import Pair

class Calibration(object):
    """Calibration is an abstract class defining a relationship between the numeric values
    in an image array and physical quantities.

    Properties:
        * name (str): The name of the quantity that this Calibration converts to.
        * factor (Scalar): The value or array that multiplies DN values.
        * baseline (Scalar): An offset value subtracted from each DN before the factor is
          applied.
        * shape (tuple): The broadcasted shape of the factor and the baseline. When
          applying the Calibration to a data object, the data object, excluding spatial
          indices, must be broadcastable to this shape.
        * fov (FOV or None): The FOV object to which this calibration refers; None if the
          object does not require an FOV.
    """

    ######################################################################################
    # Methods to be defined for each Calibration subclass
    ######################################################################################

    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar or array-like): Un-calibrated image array values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image. Note the dn
                and uv_pair will be casted to the same shape.

        Returns:
            (Scalar): Calibrated values.
        """

        raise NotImplementedError(type(self).__name__ + '.extended_from_dn '
                                  'is not implemented')

    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Parameters:
            value (Scalar or array-like): Calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image. Note the dn
                and uv_pair will be casted to the same shape.

        Returns:
            An object of the same class and shape as value, but containing the
                uncalibrated DN values.
        """

        raise NotImplementedError(type(self).__name__ + '.dn_from_extended '
                                  'is not implemented')

    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar or array-like): Un-calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image. Note the dn
                and uv_pair will be casted to the same shape.

        Returns:
            (Scalar): Calibrated values.
        """

        raise NotImplementedError(type(self).__name__ + '.point_from_dn '
                                  'is not implemented')

    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Parameters:
            value (Scalar or array-like): Calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image. Note the dn
                and uv_pair will be casted to the same shape.

        Returns:
            An object of the same class and shape as value, but containing the
                uncalibrated DN values.
        """

        raise NotImplementedError(type(self).__name__ + '.dn_from_extended '
                                  'is not implemented')

    def prescale(self, factor, baseline=0., name=''):
        """A version of this Calibration in which image DNs are re-scaled before the
        calibration is applied.

        Parameters:
            factor (float): Scale factor to apply to DN values.
            baseline (float, optional): An optional baseline value to subtract from every
                DN value before applying the new scale factor.
            name (str, optional): Optional new name. If blank, the existing name is
                preserved.

        Returns:
            A new object with the given scale factor and baseline incorporated.
        """

        raise NotImplementedError(type(self).__name__ + '.prescale '
                                  'is not implemented')

    ######################################################################################
    # Methods probably not requiring overrides
    ######################################################################################

    def value_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        DEPRECATED. Use extended_from_dn or point_from_dn.

        Parameters:
            dn (Scalar or array-like): Un-calibrated image array values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image. Note the dn
                and uv_pair will be casted to the same shape.

        Returns:
            (Scalar): Calibrated values.
        """

        return self.extended_from_dn(dn, uv_pair)

    def dn_from_value(self, value, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        DEPRECATED. Use dn_from_extended or dn_from_point.

        Parameters:
            value (Scalar or array-like): Calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image. Note the dn
                and uv_pair will be casted to the same shape.

        Returns:
            An object of the same class and shape as value, but containing the
                uncalibrated DN values.
        """

        return self.dn_from_extended(value, uv_pair)

    ######################################################################################
    # Support methods
    ######################################################################################

    def area_factor(self, uv_pair):
        """Relative pixel area relative to the center of the field of view.

        Requires that the class have an attribute "fov", containing either the FOV object
        or an area map.

        Parameters:
            uv_pair (Pair): (u,v) indices into the image.

        Returns:
            Area factors.
        """

        if isinstance(self.fov, np.ndarray):
            uv_pair = Pair.as_pair(uv_pair, recursive=False)
            uv = uv_pair.int(self.fov.shape, clip=True)
            return self.fov[uv.vals[...,0], uv.vals[...,1]]

        return self.fov.area_factor(uv_pair)

##########################################################################################
