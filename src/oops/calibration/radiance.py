##########################################################################################
# oops/calibration/radiance.py
##########################################################################################

import numpy as np

from polymath                   import Scalar, Qube
from oops.calibration.flatcalib import FlatCalib


class Radiance(FlatCalib):
    """A calibration for an image array in units of radiance within a distorted FOV.

    Radiance values are always scaled to the pixel area, so a uniform source will appear
    as an array of uniform values.
    """

    def __init__(self, name, fov, factor, baseline=0.):
        """Constructor for a Radiance Calibration.

        Parameters:
            name (str): The name of the value returned by the calibration, e.g.,
                "REFLECTIVITY".
            fov (FOV): The field of view, used to model the distortion. Alternatively, it
                can be a 2-D array containing the pixel area corrections.
            factor (Scalar): A constant scale factor to be applied to every pixel in the
                field of view.
            baseline (Scalar, optional): An optional baseline value to subtract from the
                image before applying the scale factor. Note that the factor and baseline
                values could be arrays for cases in which the non-spatial axes of the data
                array require different scalings. Their shapes must broadcast to the shape
                of the data array after the spatial axes are eliminated.
        """

        self.name = name
        self.fov = fov

        factor = Scalar.as_scalar(factor)
        baseline = Scalar.as_scalar(baseline)
        self.has_baseline = np.any(baseline.vals != 0)

        (self.factor, self.baseline) = Qube.broadcast(factor, baseline)
        self.shape = self.factor.shape

    def __getstate__(self):
        return (self.name, self.fov, self.factor, self.baseline)

    def __setstate__(self, state):
        self.__init__(*state)

    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar): Un-calibrated image array values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values for a point source.
        """

        (uv_pair, factor, baseline) = self._factor_and_baseline(uv_pair)
        dn = Scalar.as_scalar(dn)
        if self.has_baseline:
            dn = dn - baseline

        return factor * dn * self.area_factor(uv_pair)

    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Un-calibrated values for a point source.
        """

        (uv_pair, factor, baseline) = self._factor_and_baseline(uv_pair)
        dn = Scalar.as_scalar(value) / (factor * self.area_factor(uv_pair))
        if self.has_baseline:
            dn += baseline

        return dn

    def prescale(self, factor, baseline=0., *, name=''):
        """A version of this Calibration with image DNs re-scaled beforehand.

        Parameters:
            factor (Scalar): Scale factor to apply to DN values.
            baseline (Scalar, optional): An optional baseline value to subtract from every
                DN value before applying the new scale factor.
            name (str, optional): Optional new name. If blank, the existing name is
                preserved.

        Returns:
            Calibration: A new object with the given `factor` and `baseline` incorporated.
        """

        (name, factor, baseline) = self._prescaled_args(factor, baseline, name=name)
        return Radiance(name, self.fov, factor, baseline)

##########################################################################################
