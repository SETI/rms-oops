##########################################################################################
# oops/calibration/rawcounts.py
##########################################################################################

import numpy as np

from polymath                   import Scalar, Qube
from oops.calibration.flatcalib import FlatCalib


class RawCounts(FlatCalib):
    """A Calibration subclass for an image array of raw photon counts.

    When viewing a source of uniform brightness in a distorted FOV, the raw counts tend to
    be larger where the pixel areas are larger.
    """

    def __init__(self, name, fov, factor, baseline=0.):
        """Constructor for a RawCounts Calibration.

        Parameters:
            name (str): The name of the value returned by the calibration, e.g.,
                "REFLECTIVITY".
            fov (FOV): The field of view, used to model the distortion. Alternatively, it
                can be a 2-D array containing the pixel area corrections.
            factor (float): A constant scale factor to be applied to every pixel in the
                field of view.
            baseline (float, optional): An optional baseline value to subtract from the
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

    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar): Un-calibrated image array values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values for an extended source.
        """

        (uv_pair, factor, baseline) = self.factor_and_baseline(uv_pair)

        if self.has_baseline:
            dn = dn - baseline

        return dn * factor / self.area_factor(uv_pair)

    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated `(u,v)` pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Un-calibrated values for an extended source.
        """

        (uv_pair, factor, baseline) = self.factor_and_baseline(uv_pair)

        dn = value * self.area_factor(uv_pair) / factor

        if self.has_baseline:
            dn += baseline

        return dn

    def prescale(self, factor, baseline=0., *, name=''):
        """A version of this Calibration in which image DNs are re-scaled before the
        calibration is applied.

        Parameters:
            factor (float): Scale factor to apply to DN values.
            baseline (float, optional): An optional baseline value to subtract from every
                DN value before applying the new scale factor.
            name (str, optional): Optional new name. If blank, the existing name is
                preserved.

        Returns:
            Calibration: A new object with the given `factor` and `baseline` incorporated.
        """

        (name, factor, baseline) = self.prescaled_args(factor, baseline, name=name)
        return RawCounts(name, self.fov, factor, baseline)

##########################################################################################
