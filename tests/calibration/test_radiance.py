################################################################################
# tests/calibration/test_radiance.py
################################################################################

import unittest
import numpy as np

from polymath         import Pair, Scalar
from oops.calibration import Radiance
from oops.fov         import FlatFOV
from oops.constants   import RPD
from oops.config      import AREA_FACTOR


class Test_Radiance(unittest.TestCase):

  def runTest(self):

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
