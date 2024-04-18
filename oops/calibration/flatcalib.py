################################################################################
# oops/calibration/flatcalib.py: Subclass FlatCalib of class Calibration
################################################################################

import numpy as np

from polymath         import Scalar, Pair, Qube
from oops.calibration import Calibration

class FlatCalib(Calibration):
    """Calibration subclass to use for an un-distorted field of view."""

    def __init__(self, name, factor, baseline=0., fov=None):
        """Constructor for a FlatCalib object.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            factor      a scale factor to be applied to every pixel in the field
                        of view.
            baseline    an optional baseline value to subtract from the image
                        before applying the scale factor.
            fov         ignored by FlatCalib. Provided for compatibility with
                        subclasses Radiance and RawCounts.

            Note that the factor and baseline values could be arrays for cases
            in which the non-spatial axes of the data array require different
            scalings. Their shapes must broadcast to the shape of the data array
            after the spatial axes are removed.
        """

        self.name = name

        factor = Scalar.as_scalar(factor)
        baseline = Scalar.as_scalar(baseline)
        self.has_baseline = np.any(baseline.vals != 0)

        (self.factor, self.baseline) = Qube.broadcast(factor, baseline)
        self.shape = self.factor.shape
        self.fov = None

    def __getstate__(self):
        return (self.name, self.factor, self.baseline)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Input:
            dn          a Scalar or array of un-calibrated image array values at
                        the given pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image.

        Return:         calibrated values.
        """

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            factor = self.factor[indx]
            baseline = self.baseline[indx]
        else:
            factor = self.factor
            baseline = self.baseline

        if self.has_baseline:
            return (dn - baseline) * factor

        return dn * factor

    #===========================================================================
    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Input:
            value       a Scalar or array of calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image.

        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            factor = self.factor[indx]
            baseline = self.baseline[indx]
        else:
            factor = self.factor
            baseline = self.baseline

        if self.has_baseline:
            return value / factor + baseline

        return value / factor

    #===========================================================================
    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Input:
            dn          a Scalar or array of un-calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image.

        Return:         calibrated values.
        """

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            factor = self.factor[indx]
            baseline = self.baseline[indx]
        else:
            factor = self.factor
            baseline = self.baseline

        if self.has_baseline:
            return (dn - baseline) * factor

        return dn * factor

    #===========================================================================
    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Input:
            value       a Scalar or array of calibrated values at the given
                        pixel coordinates.
            uv_pair     associated (u,v) pixel coordinates in the image.

        Return:         an object of the same class and shape as value, but
                        containing the uncalibrated DN values.
        """

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            factor = self.factor[indx]
            baseline = self.baseline[indx]
        else:
            factor = self.factor
            baseline = self.baseline

        if self.has_baseline:
            return value / factor + baseline

        return value / factor

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

        # new_dn = factor * (dn - baseline)
        #
        # value = self.factor * (dn - self.baseline)
        #   = self.factor * (factor * (dn - baseline) - self.baseline)
        #   = (self.factor*factor) * (dn - baseline - self.baseline/factor)
        #
        # new_factor = self.factor * factor
        # new_baseline = baseline + self.baseline/factor

        return FlatCalib(name or self.name,
                         factor = factor * self.factor,
                         baseline = baseline + self.baseline/factor)

################################################################################
