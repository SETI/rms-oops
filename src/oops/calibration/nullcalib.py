##########################################################################################
# oops/calibration/nullcalib.py
##########################################################################################

from polymath                   import Scalar
from oops.calibration           import Calibration
from oops.calibration.flatcalib import FlatCalib


class NullCalib(Calibration):
    """Calibration subclass that leaves data values unchanged."""

    def __init__(self, name):
        """Constructor for a NullCalib object.

        Parameters:
            name (str): The name of the value returned by the calibration, e.g.,
                "REFLECTIVITY".
        """

        self.name = name

        # Required attributes
        self.factor = Scalar.ONE
        self.baseline = Scalar.ZERO
        self.has_baseline = False
        self.fov = None
        self.shape = ()

    def __getstate__(self):
        return (self.name,)

    def __setstate__(self, state):
        self.__init__(*state)

    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar): Un-calibrated image array values at the given pixel coordinates.
            uv_pair (Pair): Associated *(u,v)* pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values for an extended source.
        """

        return Scalar.as_scalar(dn)

    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated *(u,v)* pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Un-calibrated values for an extended source.
        """

        return Scalar.as_scalar(value)

    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar): Un-calibrated image array values at the given pixel coordinates.
            uv_pair (Pair): Associated *(u,v)* pixel coordinates in the image. Note that
                `dn` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Calibrated values for a point source.
        """

        return Scalar.as_scalar(dn)

    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Parameters:
            value (Scalar): Calibrated values at the given pixel coordinates.
            uv_pair (Pair): Associated *(u,v)* pixel coordinates in the image. Note that
                `value` and `uv_pair` will be casted to the same shape.

        Returns:
            Scalar: Un-calibrated values for a point source.
        """

        return Scalar.as_scalar(value)

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

        # Pre-scaling requires a FlatCalib instead
        return FlatCalib(name or self.name, factor, baseline)

##########################################################################################
