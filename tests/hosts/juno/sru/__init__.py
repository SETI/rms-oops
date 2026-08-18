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

    #===========================================================================
    def runTest(self):
        pass


##############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
