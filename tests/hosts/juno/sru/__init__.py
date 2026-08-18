################################################################################
# tests/hosts/juno/sru/__init__.py
################################################################################
import unittest
import numpy as np

from polymath import Pair
from oops.hosts.juno.sru import SRU


#===============================================================================
class Test_Juno_SRU_FOV(unittest.TestCase):
    """Validate the SRU FOV against the distortion formulas in the SIS,
    JUNO_SRU_EDR_CRT_SIS_V01_2 section 5.4.1. These tests require no SPICE
    kernels or data files.
    """

    #===========================================================================
    def runTest(self):
        fov = SRU.fov()
        (a0, a1, a2, a3) = SRU.DISTORTION

        # The boresight pixel maps to the optic axis
        xy = fov.xy_from_uv(SRU.UV_LOS)
        self.assertTrue(abs(xy.vals[0]) < 1.e-15)
        self.assertTrue(abs(xy.vals[1]) < 1.e-15)

        # Spot-check the SIS distortion formula: pixel (row,col) has pinhole
        # tangents (row-255.5, col-255.5)/fl, scaled radially by f(R); the
        # camera-frame x axis lies along increasing sample (col, the SIS x
        # direction) and y along increasing line (row, the SIS y direction).
        for (row, col) in [(0., 2.), (509., 511.), (100., 400.), (255.5, 511.)]:
            tanx = (col - 255.5)/SRU.FL_PIXELS
            tany = (row - 255.5)/SRU.FL_PIXELS
            R = np.sqrt(tanx**2 + tany**2)
            f = a0 + a1*R + a2*R**2 + a3*R**4
            xy = fov.xy_from_uv((col, row))
            self.assertTrue(abs(xy.vals[0] - f*tanx) < 1.e-12)
            self.assertTrue(abs(xy.vals[1] - f*tany) < 1.e-12)

        # uv -> xy -> uv round trip at sub-pixel precision
        uv = Pair(np.random.RandomState(0).uniform(0., 512., (100,2)))
        uv2 = fov.uv_from_xy(fov.xy_from_uv(uv))
        self.assertTrue(np.abs(uv2.vals - uv.vals).max() < 1.e-6)

        # Full field of view is 16.4 degrees square per the SIS
        corner = fov.xy_from_uv((0., 0.))
        half_diag = np.degrees(np.arctan(np.hypot(*corner.vals)))
        self.assertTrue(abs(half_diag - 16.4/2.*np.sqrt(2.)) < 0.15)


#===============================================================================
class Test_Juno_SRU(unittest.TestCase):
    """Regression tests for from_file() using a real EDR from the shared test
    data tree. Skipped if the test data or SPICE kernels are unavailable.
    """

    DATA = 'juno/sru/SRU_1_2024100T045333_60_V01.FIT'

    #===========================================================================
    def runTest(self):
        import julian
        import oops
        import oops.hosts.juno.sru as sru
        from oops.unittester_support import TEST_DATA_PREFIX

        from polymath import Vector3

        datspec = TEST_DATA_PREFIX / self.DATA
        try:
            datspec.retrieve()
            obs = sru.from_file(datspec)
        except (FileNotFoundError, OSError) as e:
            self.skipTest('SRU test data or kernels unavailable: ' + str(e))

        # Metadata extraction
        self.assertEqual(obs.instrument, 'SRU1')
        self.assertEqual(obs.target, 'IO')
        self.assertTrue(obs.tdi_on)
        self.assertEqual(obs.texp, 0.01)
        tstart = julian.tdb_from_tai(julian.tai_from_iso('2024-04-09T04:53:33.955'))
        self.assertTrue(abs(obs.tstart - tstart) < 1.e-6)

        # FITS data array in (line, sample) order; dummy columns 0-1 and rows
        # 510-511 are zero
        self.assertEqual(obs.data.shape, (512, 512))
        self.assertTrue(np.all(obs.data[:,0:2] == 0))
        self.assertTrue(np.all(obs.data[510:512,:] == 0))
        self.assertTrue(obs.data.max() > 0)

        # Detached-label resolution: the .LBL path yields the same observation
        obs2 = sru.from_file(datspec.with_suffix('.LBL'))
        self.assertTrue(np.all(obs2.data == obs.data))
        self.assertEqual(obs2.tstart, obs.tstart)

        # The camera frame is inertially frozen at START_TIME...
        xform0 = obs.frame.wrt(oops.frame.Frame.J2000).transform_at_time(obs.tstart)
        xform1 = obs.frame.wrt(oops.frame.Frame.J2000).transform_at_time(obs.tstart + 1000.)
        self.assertTrue(np.all(xform0.matrix.vals == xform1.matrix.vals))

        # ...and the boresight there matches the label RA/DEC
        bore = xform0.unrotate(Vector3((0., 0., 1.))).vals
        ra = np.degrees(np.arctan2(bore[1], bore[0])) % 360.
        dec = np.degrees(np.arcsin(bore[2]))
        self.assertTrue(abs(ra - obs.dict['RIGHT_ASCENSION']) < 0.01)
        self.assertTrue(abs(dec - obs.dict['DECLINATION']) < 0.01)

        # Distinct observations own distinct frame objects
        self.assertIsNot(obs2.frame, obs.frame)


##############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
