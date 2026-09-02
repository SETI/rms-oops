##########################################################################################
# oops/calibration/calibration_.py
##########################################################################################

import numpy as np
from polymath import Pair


class Calibration(object):
    """Calibration is an abstract class defining a relationship between the numeric values
    in an image array and physical quantities.

    Properties:
        name (str): The name of the quantity that this Calibration converts to.
        factor (Scalar): The value or array that multiplies DN values.
        baseline (Scalar): An offset value subtracted from each DN before the factor is
            applied.
        has_baseline (bool): True if this object has a non-zero baseline.
        shape (tuple): The broadcasted shape of the factor and the baseline. When
            applying the Calibration to a data object, the data object, excluding spatial
            indices, must be broadcastable to this shape.
        fov (FOV or None): The FOV object to which this calibration refers; None if the
            object does not require an FOV.
    """

    ######################################################################################
    # Methods to be defined for each Calibration subclass
    ######################################################################################

    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar): Un-calibrated image array values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values for an extended source.
        """

        raise NotImplementedError(f'{type(self).__name__}.extended_from_dn is not '
                                  'implemented')

    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Un-calibrated values for an extended source.
        """

        raise NotImplementedError(f'{type(self).__name__}.dn_from_extended is not '
                                  'implemented')

    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar): Un-calibrated image array values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values for a point source.
        """

        raise NotImplementedError(f'{type(self).__name__}.point_from_dn is not '
                                  'implemented')

    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Un-calibrated values for a point source.
        """

        raise NotImplementedError(f'{type(self).__name__}.dn_from_point is not '
                                  'implemented')

    def prescale(self, factor, baseline=0., *, name=''):
        """A version of this Calibration in which image DNs are re-scaled before the
        calibration is applied.

        Parameters:
            factor (np.ndarray or float): Scale factor to apply to DN values.
            baseline (float, optional): An optional baseline value to subtract from every
                DN value before applying the new scale factor.
            name (str, optional): Optional new name. If blank, the existing name is
                preserved.

        Returns:
            Calibration: A new object with the given `factor` and `baseline` incorporated.
        """

        raise NotImplementedError(f'{type(self).__name__}.prescale is not implemented')

    ######################################################################################
    # Methods probably not requiring overrides
    ######################################################################################

    def value_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        DEPRECATED. Use extended_from_dn or point_from_dn.

        Parameters:
            dn (Scalar or array-like): Un-calibrated image array values at the given pixel
                coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values.
        """

        return self.extended_from_dn(dn, uv_pair)

    def dn_from_value(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        DEPRECATED. Use dn_from_extended or dn_from_point.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: The uncalibrated DN values.
        """

        return self.dn_from_extended(value, uv_pair)

    ######################################################################################
    # Support methods
    ######################################################################################

    def factor_and_baseline(self, uv_pair):
        """The factor and baseline, shaped to broadcast against the given coordinates.

        The factor and the baseline describe the non-spatial axes of a data array, so
        each needs a new trailing axis for every axis of the pixel coordinates before the
        two can be combined.

        Parameters:
            uv_pair (Pair): `(u,v)` pixel coordinates in the image.

        Returns:
            tuple[Pair, Scalar, Scalar]: The pixel coordinates converted to a Pair,
            followed by the factor and the baseline, each shaped to broadcast against
            those coordinates.
        """

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            return (uv_pair, self.factor[indx], self.baseline[indx])

        return (uv_pair, self.factor, self.baseline)

    def prescaled_args(self, factor, baseline=0., *, name=''):
        """The name, factor and baseline of a pre-scaled version of this Calibration.

        This performs the algebra shared by every `prescale` implementation. The caller
        supplies the results to its own constructor.

        Parameters:
            factor (np.ndarray or float): Scale factor to apply to DN values.
            baseline (float, optional): An optional baseline value to subtract from every
                DN value before applying the new scale factor.
            name (str, optional): Optional new name. If blank, the existing name is
                preserved.

        Returns:
            tuple[str, Scalar, Scalar]: The name, factor and baseline describing this
            Calibration applied to DN values that have already been re-scaled.
        """

        # new_dn = factor * (dn - baseline)
        #
        # value = self.factor * (new_dn - self.baseline)
        #   = self.factor * (factor * (dn - baseline) - self.baseline)
        #   = (self.factor*factor) * (dn - baseline - self.baseline/factor)
        #
        # new_factor = self.factor * factor
        # new_baseline = baseline + self.baseline/factor

        return (name or self.name,
                factor * self.factor,
                baseline + self.baseline/factor)

    def area_factor(self, uv_pair):
        """Pixel area relative to the center of the field of view.

        Requires that the class have an attribute "fov", containing either the FOV object
        or an area map.

        Parameters:
            uv_pair (Pair): `(u,v)` indices into the image.

        Returns:
            Scalar: Area factors.
        """

        if isinstance(self.fov, np.ndarray):
            uv_pair = Pair.as_pair(uv_pair, recursive=False)
            uv = uv_pair.int(self.fov.shape, clip=True)
            return self.fov[uv.vals[...,0], uv.vals[...,1]]

        return self.fov.area_factor(uv_pair)

##########################################################################################
