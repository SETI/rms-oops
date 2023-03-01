################################################################################
# oops/calibration/radiance.py: Radiance subclass of Calibration
################################################################################

import numpy as np
from polymath import Scalar, Pair, Qube
from oops.calibration.flatcalib import FlatCalib

class Radiance(FlatCalib):
    """A Calibration subclass for an image array in units of radiance within a
    distorted FOV.

    Radiance values are always scaled to the pixel area, so a uniform source
    will appear as an array of uniform values.
    """

    def __init__(self, name, fov, factor, baseline=0.):
        """Constructor for a RawCounts Calibration.

        Input:
            name        the name of the value returned by the calibration, e.g.,
                        "REFLECTIVITY".
            fov         the field of view, used to model the distortion.
                        Alternatively, it can be a 2-D array containing the
                        pixel area corrections.
            factor      a constant scale factor to be applied to every pixel in
                        the field of view.
            baseline    an optional baseline value to subtract from the image
                        before applying the scale factor.

            Note that the factor and baseline values could be arrays for cases
            in which the non-spatial axes of the data array require different
            scalings. Their shapes must broadcast to the shape of the data array
            after the spatial axes are eliminated.
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

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            factor = self.factor[indx]
            baseline = self.baseline[indx]
        else:
            factor = self.factor
            baseline = self.baseline

        if self.has_baseline:
            dn = dn - baseline

        return factor * dn * self.area_factor(uv_pair)

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

        uv_pair = Pair.as_pair(uv_pair)

        if uv_pair.shape and self.shape:
            indx = (Ellipsis,) + len(uv_pair.shape) * (None,)
            factor = self.factor[indx]
            baseline = self.baseline[indx]
        else:
            factor = self.factor
            baseline = self.baseline

        dn = value / (factor * self.area_factor(uv_pair))

        if self.has_baseline:
            dn += baseline

        return dn

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

        return Radiance(name or self.name,
                        fov = self.fov,
                        factor = factor * self.factor,
                        baseline = baseline + self.baseline/factor)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Radiance(unittest.TestCase):

  def runTest(self):

    import numpy as np
    from polymath import Pair
    from oops.fov.flatfov import FlatFOV
    from oops.constants import RPD
    from oops.config import AREA_FACTOR

    try:
        AREA_FACTOR.old = True

        flat_fov = FlatFOV((RPD/3600.,RPD/3600.), (1024,1024))
        cal = Radiance('TEST', flat_fov, 5.)
        self.assertEqual(cal.extended_from_dn(0., (512,512)), 0.)
        self.assertEqual(cal.extended_from_dn(0., (10,10)), 0.)
        self.assertEqual(cal.extended_from_dn(5., (512,512)), 25.)
        self.assertEqual(cal.extended_from_dn(5., (10,10)), 25.)
        self.assertEqual(cal.extended_from_dn(.5, (512,512)), 2.5)
        self.assertEqual(cal.extended_from_dn(.5, (10,10)), 2.5)

        self.assertEqual(cal.dn_from_extended(0., (512,512)), 0.)
        self.assertEqual(cal.dn_from_extended(0., (10,10)), 0.)
        self.assertEqual(cal.dn_from_extended(25., (512,512)), 5.)
        self.assertEqual(cal.dn_from_extended(25., (10,10)), 5.)
        self.assertEqual(cal.dn_from_extended(2.5, (512,512)), .5)
        self.assertEqual(cal.dn_from_extended(2.5, (10,10)), .5)

        self.assertEqual(cal.point_from_dn(0., (512,512)), 0.)
        self.assertEqual(cal.point_from_dn(0., (10,10)), 0.)
        self.assertEqual(cal.point_from_dn(5., (512,512)), 25.)
        self.assertEqual(cal.point_from_dn(5., (10,10)), 25.)
        self.assertEqual(cal.point_from_dn(.5, (512,512)), 2.5)
        self.assertEqual(cal.point_from_dn(.5, (10,10)), 2.5)

        self.assertEqual(cal.dn_from_point(0., (512,512)), 0.)
        self.assertEqual(cal.dn_from_point(0., (10,10)), 0.)
        self.assertEqual(cal.dn_from_point(25., (512,512)), 5.)
        self.assertEqual(cal.dn_from_point(25., (10,10)), 5.)
        self.assertEqual(cal.dn_from_point(2.5, (512,512)), .5)
        self.assertEqual(cal.dn_from_point(2.5, (10,10)), .5)

        a = Scalar(np.arange(10000).reshape((100,100)))
        self.assertEqual(a, cal.dn_from_extended(cal.extended_from_dn(a, (10,10)), (10,10)))
        self.assertEqual(a, cal.dn_from_point(cal.point_from_dn(a, (10,10)), (10,10)))

        cal = Radiance('TEST', flat_fov, 5., 1.)
        self.assertEqual(cal.extended_from_dn(1., (512,512)), 0.)
        self.assertEqual(cal.extended_from_dn(1., (10,10)), 0.)
        self.assertEqual(cal.extended_from_dn(6., (512,512)), 25.)
        self.assertEqual(cal.extended_from_dn(6., (10,10)), 25.)
        self.assertEqual(cal.extended_from_dn(1.5, (512,512)), 2.5)
        self.assertEqual(cal.extended_from_dn(1.5, (10,10)), 2.5)

        self.assertEqual(cal.dn_from_extended(0., (512,512)), 1.)
        self.assertEqual(cal.dn_from_extended(0., (10,10)), 1.)
        self.assertEqual(cal.dn_from_extended(25., (512,512)), 6.)
        self.assertEqual(cal.dn_from_extended(25., (10,10)), 6.)
        self.assertEqual(cal.dn_from_extended(2.5, (512,512)), 1.5)
        self.assertEqual(cal.dn_from_extended(2.5, (10,10)), 1.5)

        self.assertEqual(cal.point_from_dn(1., (512,512)), 0.)
        self.assertEqual(cal.point_from_dn(1., (10,10)), 0.)
        self.assertEqual(cal.point_from_dn(6., (512,512)), 25.)
        self.assertEqual(cal.point_from_dn(6., (10,10)), 25.)
        self.assertEqual(cal.point_from_dn(1.5, (512,512)), 2.5)
        self.assertEqual(cal.point_from_dn(1.5, (10,10)), 2.5)

        self.assertEqual(cal.dn_from_point(0., (512,512)), 1.)
        self.assertEqual(cal.dn_from_point(0., (10,10)), 1.)
        self.assertEqual(cal.dn_from_point(25., (512,512)), 6.)
        self.assertEqual(cal.dn_from_point(25., (10,10)), 6.)
        self.assertEqual(cal.dn_from_point(2.5, (512,512)), 1.5)
        self.assertEqual(cal.dn_from_point(2.5, (10,10)), 1.5)

        a = Scalar(np.arange(10000).reshape((100,100)))
        self.assertEqual(a, cal.dn_from_extended(cal.extended_from_dn(a, (10,10)), (10,10)))
        self.assertEqual(a, cal.dn_from_point(cal.point_from_dn(a, (10,10)), (10,10)))

        # fov[0,0] = 1; fov[9,9] = 1.125
        fov = 1 + np.arange(100).reshape((10,10))/1000
        fov[9,9] = 1.125

        uv = Pair([(0,0),(9,9)])
        dn = np.array([2,3])

        # values = 5 * dn
        cal = Radiance('CAL', fov, 5.)
        values = cal.extended_from_dn(dn, uv)
        self.assertEqual(values[0], 5*2)
        self.assertEqual(values[1], 5*3)

        dn2 = cal.dn_from_extended(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        values = cal.point_from_dn(dn, uv)
        self.assertEqual(values[0], 10)
        self.assertEqual(values[1], 15 * fov[9,9])

        dn2 = cal.dn_from_point(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        # values = 5 * (dn - 1)
        cal = Radiance('CAL', fov, 5., baseline=1.)
        values = cal.extended_from_dn(dn, uv)
        self.assertEqual(values[0], 5)
        self.assertEqual(values[1], 10)

        dn2 = cal.dn_from_extended(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        values = cal.point_from_dn(dn, uv)
        self.assertEqual(values[0], 5)
        self.assertEqual(values[1], 10 * fov[9,9])

        dn2 = cal.dn_from_point(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        # values = 5 * ((4*dn) - 1)
        cal2 = cal.prescale(4, name='X4')
        self.assertEqual(cal2.name, 'X4')

        values = cal2.extended_from_dn(dn, uv)
        self.assertEqual(values[0], 5*(8-1))
        self.assertEqual(values[1], 5*(12-1))

        dn2 = cal2.dn_from_extended(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        values = cal2.point_from_dn(dn, uv)
        self.assertEqual(values[0], 5*(8-1))
        self.assertEqual(values[1], 5*(12-1) * fov[9,9])

        dn2 = cal2.dn_from_point(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        # values = 5 * ((4*(dn - 1)) - 1)
        cal2 = cal.prescale(4,1)
        self.assertEqual(cal2.name, cal.name)

        values = cal2.extended_from_dn(dn, uv)
        self.assertEqual(values[0], 5*(4-1))
        self.assertEqual(values[1], 5*(8-1))

        dn2 = cal2.dn_from_extended(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        values = cal2.point_from_dn(dn, uv)
        self.assertEqual(values[0], 5*(4-1))
        self.assertEqual(values[1], 5*(8-1) * fov[9,9])

        dn2 = cal2.dn_from_point(values, uv)
        self.assertEqual(dn[0], dn2[0])
        self.assertEqual(dn[1], dn2[1])

        # Alternative shape:
        # Data array has shape (3,10,10); new scale factor has shape (3,)
        # Scale factors are [1,2,4]
        # values = 5 * (([1,2,4]*(dn - 1)) - 1)
        cal2 = cal.prescale([1,2,4],1)
        factors = Scalar([1,2,4])
        dn = np.array(3*[[2,3]])

        values = cal2.extended_from_dn(dn, uv)
        self.assertEqual(values.shape, (3,2))
        self.assertEqual(values[:,0], 5*factors*(2-1) - 5)
        self.assertEqual(values[:,1], 5*factors*(3-1) - 5)

        dn2 = cal2.dn_from_extended(values, uv)
        self.assertEqual(dn, dn2)

        values = cal2.point_from_dn(dn, uv)
        self.assertEqual(values[:,0], 5*factors*(2-1) - 5)
        self.assertEqual(values[:,1], (5*factors*(3-1) - 5) * fov[9,9])

        dn2 = cal2.dn_from_point(values, uv)
        self.assertEqual(dn, dn2)

    finally:
        AREA_FACTOR.old = False

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
