################################################################################
# tests/frame/test_spiceframe.py
################################################################################

import numpy as np
import unittest

import cspyce

from polymath       import Scalar, Vector3
from oops.config    import QUICK
from oops.constants import DPR
from oops.event     import Event
from oops.frame     import Frame, SpiceFrame, QuickFrame
from oops.path      import Path
from oops.path.spicepath import SpicePath

from oops.unittester_support import TEST_SPICE_PFX


class Test_SpiceFrame(unittest.TestCase):

    def setUp(self):
        paths = TEST_SPICE_PFX.retrieve(['naif0009.tls', 'pck00010.tpc',
                                         'de421.bsp'])
        for path in paths:
            cspyce.furnsh(path)
        Path.USE_QUICKPATHS = False
        Frame.USE_QUICKFRAMES = False
        Path.reset_registry()
        Frame.reset_registry()

    def tearDown(self):
        Path.reset_registry()
        Frame.reset_registry()
        Path.USE_QUICKPATHS = True
        Frame.USE_QUICKFRAMES = True

    def runTest(self):

        np.random.seed(6242)

        _ = SpicePath('EARTH', 'SSB')

        _ = SpiceFrame('IAU_EARTH', 'J2000')
        time  = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6,1)
        event = Event(time, (posvel[...,0:3,0],posvel[...,3:6,0]), 'SSB',
                                                                   'J2000')
        rotated = event.wrt_frame('IAU_EARTH')

        for i,t in np.ndenumerate(time.vals):
            matrix6 = cspyce.sxform('J2000', 'IAU_EARTH', t)
# The matrix subclass is to be deprecated? Really?
#             spiceval = np.matrix(matrix6) * np.matrix(posvel[i])
            spiceval = np.matmul(matrix6, posvel[i])[...,np.newaxis]

            dpos = rotated.pos[i].vals[...,np.newaxis] - spiceval[0:3,0]
            dvel = rotated.vel[i].vals[...,np.newaxis] - spiceval[3:6,0]

            self.assertTrue(np.all(np.abs(dpos) < 1.e-15))
            self.assertTrue(np.all(np.abs(dvel) < 1.e-15))

        # Tests of combined frames
        Path.reset_registry()
        Frame.reset_registry()

        _ = SpicePath('EARTH', 'SSB')
        _ = SpicePath('VENUS', 'EARTH')
        _ = SpicePath('MARS', 'VENUS')
        _ = SpicePath('MOON', 'VENUS')

        _ = SpiceFrame('IAU_EARTH', 'J2000')
        _ = SpiceFrame('B1950', 'IAU_EARTH')
        _ = SpiceFrame('IAU_VENUS', 'B1950')
        _ = SpiceFrame('IAU_MARS', 'J2000')
        _ = SpiceFrame('IAU_MOON', 'B1950')

        times = Scalar(np.arange(-3.e8, 3.01e8, 0.5e7))

        frame = Frame.as_frame('IAU_EARTH').wrt('J2000')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('J2000', 'IAU_EARTH', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = Frame.as_frame('J2000').wrt('IAU_EARTH')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('IAU_EARTH', 'J2000', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = Frame.as_frame('B1950').wrt('J2000')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('J2000', 'B1950', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = Frame.as_frame('J2000').wrt('B1950')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('B1950', 'J2000', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        Path.reset_registry()
        Frame.reset_registry()

        ########################################
        # Test for a Cassini C kernel
        ########################################

        # Load all the required kernels for Cassini ISS on 2007-312
        paths = TEST_SPICE_PFX.retrieve(['naif0009.tls', 'cas00149.tsc',
                                         'cas_v40.tf','cas_status_v04.tf',
                                         'cas_iss_v10.ti', 'pck00010.tpc',
                                         'cpck14Oct2011.tpc', 'de421.bsp',
                                         'sat052.bsp', 'sat083.bsp',
                                         'sat125.bsp', 'sat128.bsp',
                                         'sat164.bsp', '07312_07317ra.bc',
                                         '080123R_SCPSE_07309_07329.bsp'])
        for path in paths:
            cspyce.furnsh(path)

        _ = SpicePath('CASSINI', 'SSB')
        _ = SpiceFrame('CASSINI_ISS_NAC')
        _ = SpiceFrame('CASSINI_ISS_WAC')

        # Look up N1573186009_1.IMG from COISS_2039/data/1573186009_1573197826/
        timestring = '2007-312T03:34:16.391'
        TDB = cspyce.str2et(timestring)

        nacframe = Frame.J2000.wrt('CASSINI_ISS_NAC')
        matrix = nacframe.transform_at_time(TDB).matrix
        optic_axis = (matrix * Vector3((0,0,1))).vals

        test_ra  = (np.arctan2(optic_axis[1], optic_axis[0]) * DPR) % 360
        test_dec = np.arcsin(optic_axis[2]) * DPR

        right_ascension = 194.30861     # from the index table
        declination = 3.142808

        self.assertTrue(np.all(np.abs(test_ra - right_ascension) < 0.5))
        self.assertTrue(np.all(np.abs(test_dec - declination) < 0.5))

        ########################################
        # Test of various omega methods
        ########################################

        wac1 = SpiceFrame('CASSINI_ISS_WAC', omega_type='tabulated', frame_id='wac1')
        wac2 = SpiceFrame('CASSINI_ISS_WAC', omega_type='numerical', frame_id='wac2')
        wac3 = SpiceFrame('CASSINI_ISS_WAC', omega_type='zero',      frame_id='wac3')

        # Test a single time
        xform1 = wac1.transform_at_time(TDB, quick=None)
        xform2 = wac2.transform_at_time(TDB, quick=None)
        xform3 = wac3.transform_at_time(TDB, quick=None)
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        DT = 0.2
        xform2a = wac2.transform_at_time(TDB-DT, quick=None)
        xform2b = wac2.transform_at_time(TDB+DT, quick=None)

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = (angle/(2.*DT) / xform2.omega.norm()).vals
            self.assertAlmostEqual(ratio, 1., places=9)

        # Test an array of times
        times = Scalar((TDB-5., TDB+2., TDB-9.5, TDB+7.7))

        xform1 = wac1.transform_at_time(times, quick=None)
        xform2 = wac2.transform_at_time(times, quick=None)
        xform3 = wac3.transform_at_time(times, quick=None)
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        DT = 1.
        xform2a = wac2.transform_at_time(times-DT, quick=None)
        xform2b = wac2.transform_at_time(times+DT, quick=None)

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = angle/(2.*DT) / xform2.omega.norm()
            mask = (angle.vals != 0.)
            self.assertTrue(abs(ratio[mask]).max() - 1. < 1.e-9)

        # Test a single value using transform_at_time_if_possible
        xform1 = wac1.transform_at_time_if_possible(TDB, quick=None)[1]
        xform2 = wac2.transform_at_time_if_possible(TDB, quick=None)[1]
        self.assertEqual(xform1.matrix, xform2.matrix)

        DT = 0.2
        xform2a = wac2.transform_at_time_if_possible(TDB-DT, quick=None)[1]
        xform2b = wac2.transform_at_time_if_possible(TDB+DT, quick=None)[1]

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = (angle/(2.*DT) / xform2.omega.norm()).vals
            self.assertAlmostEqual(ratio, 1., places=9)

        # Test an array of times using transform_at_time_if_possible
        times = Scalar((TDB-5., TDB+2., TDB-9.5, TDB+7.7))

        xform1 = wac1.transform_at_time_if_possible(times, quick=None)[1]
        xform2 = wac2.transform_at_time_if_possible(times, quick=None)[1]
        xform3 = wac3.transform_at_time_if_possible(times, quick=None)[1]
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        DT = 1.
        xform2a = wac2.transform_at_time_if_possible(times-DT, quick=None)[1]
        xform2b = wac2.transform_at_time_if_possible(times+DT, quick=None)[1]

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = angle/(2.*DT) / xform2.omega.norm()
            mask = (angle.vals != 0.)
            self.assertTrue(abs(ratio[mask]).max() - 1. < 1.e-9)

        # Test an array of times using transform_at_time_if_possible
        # In this run, several times will fail.
        times = Scalar((TDB-5., TDB-1.e20, TDB+2., TDB-9.5, TDB+7.7, TDB+1.e10))

        xform1 = wac1.transform_at_time_if_possible(times, quick=None)[1]
        xform2 = wac2.transform_at_time_if_possible(times, quick=None)[1]
        xform3 = wac3.transform_at_time_if_possible(times, quick=None)[1]
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        self.assertEqual(xform1.shape, (4,))    # two of the times are invalid
        self.assertEqual(xform2.shape, (4,))

        DT = 1.
        xform2a = wac2.transform_at_time_if_possible(times-DT, quick=None)[1]
        xform2b = wac2.transform_at_time_if_possible(times+DT, quick=None)[1]

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = angle/(2.*DT) / xform2.omega.norm()
            mask = (angle.vals != 0.)
            self.assertTrue(abs(ratio[mask]).max() - 1. < 1.e-9)

        ########################################
        # Tests of QuickFrame interpolation
        ########################################

        wac1 = SpiceFrame('CASSINI_ISS_WAC', omega_type='tabulated',)
        wac2 = SpiceFrame('CASSINI_ISS_WAC', omega_type='numerical',
                                             omega_dt=0.1)
        wac3 = SpiceFrame('CASSINI_ISS_WAC', omega_type='zero')

        quickdict = QUICK.dictionary.copy()
        quickdict['quickframe_numerical_omega'] = False
        quickdict['frame_time_step'] = 0.01
        wac1a = QuickFrame(wac1, (TDB-100.,TDB+100.), quickdict)

        quickdict['quickframe_numerical_omega'] = True
        wac1b = QuickFrame(wac1, (TDB-100.,TDB+100.), quickdict)

        _ = QuickFrame(wac3, (TDB-100.,TDB+100.), quickdict)

        # Test a single time
        time = TDB - 44.
        xform1 = wac1a.transform_at_time(time, quick=None)
        xform2 = wac1b.transform_at_time(time, quick=None)
        xform3 = wac2.transform_at_time(time, quick=None)
        xform4 = wac3.transform_at_time(time, quick=None)
        self.assertEqual(xform4.omega, Vector3.ZERO)

        self.assertEqual(xform1.matrix, xform2.matrix)

        diff = xform3.matrix.values - xform1.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-11)

        diff = xform4.matrix.values - xform1.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-11)

        diff = (xform3.omega - xform2.omega).norm()
        self.assertTrue(diff.vals < 1.e-7)

        diff = (xform3.omega - xform2.omega).norm() / xform2.omega.norm()
        self.assertTrue(diff.vals < 1.e-3)

        # Test the linear interpolation limit where delta-time < 1 sec
        time = Scalar((TDB - 41.0123, TDB - 41.1357, TDB - 41.6543))
        xform2 = wac1b.transform_at_time(time, quick=None)
        xform3 = wac2.transform_at_time(time, quick=None)
        xform4 = wac3.transform_at_time(time, quick=None)
        self.assertEqual(xform4.omega, Vector3.ZERO)

        diff = xform3.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 3.e-9)

        diff = xform4.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 3.e-9)

        diff = (xform3.omega - xform2.omega).norm()
        self.assertTrue(diff.max() < 1.e-7)

        # Test the linear interpolation limit where delta-time > 1 sec
        time = Scalar((TDB - 40.0123, TDB + 41.1357, TDB - 1.6543))
        xform2 = wac1b.transform_at_time(time, quick=None)
        xform3 = wac2.transform_at_time(time, quick=None)
        xform4 = wac3.transform_at_time(time, quick=None)
        self.assertEqual(xform4.omega, Vector3.ZERO)

        diff = xform3.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-9)

        diff = xform4.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-9)

        diff = (xform3.omega - xform2.omega).norm()
        self.assertTrue(diff.max() < 1.e-7)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
