##########################################################################################
# oops/calibration/nullcalib.py: Subclass NullCalib of class Calibration
##########################################################################################

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
        self.factor = 1
        self.baseline = 0
        self.fov = None
        self.shape = ()

    def __getstate__(self):
        return (self.name)

    def __setstate__(self, state):
        self.__init__(*state)

    def extended_from_dn(self, dn, uv_pair):
        """Extended-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar or array-like): Un-calibrated image array values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image.

        Returns:
            (Scalar): Calibrated values.
        """

        return dn

    def dn_from_extended(self, value, uv_pair):
        """Un-calibrated image DN from extended-source calibrated values.

        Parameters:
            value (Scalar or array-like): Calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image.

        Returns:
            An object of the same class and shape as value, but containing the
                uncalibrated DN values.
        """

        return value

    def point_from_dn(self, dn, uv_pair):
        """Point-source calibrated values for image DN and pixel coordinates.

        Parameters:
            dn (Scalar or array-like): Un-calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image.

        Returns:
            (Scalar): Calibrated values.
        """

        return dn

    def dn_from_point(self, value, uv_pair):
        """Un-calibrated image DN from point-source calibrated values.

        Parameters:
            value (Scalar or array-like): Calibrated values at the given pixel
                coordinates.
            uv_pair (Pair): Associated (u,v) pixel coordinates in the image.

        Returns:
            An object of the same class and shape as value, but containing the
                uncalibrated DN values.
        """

        return value

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

        # Pre-scaling requires a FlatCalib instead
        return FlatCalib(name or self.name, factor, baseline)

##########################################################################################
