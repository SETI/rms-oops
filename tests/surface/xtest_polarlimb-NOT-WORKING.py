################################################################################
# tests/surface/test_polarlimb.py
################################################################################

import numpy as np
import unittest

from polymath               import Vector3
from oops.constants         import HALFPI
from oops.surface.polarlimb import PolarLimb
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.spheroid  import Spheroid
from oops.surface.ellipsoid import Ellipsoid


# TODO: This test was erroneously testing class Limb instead of class PolarLimb.
# When I change Limb() to PolarLimb(), it fails. This needs to be fixed!!

class xTest_PolarLimb(unittest.TestCase):

    def runTest(self):

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.

        ground = Spheroid('SSB', 'J2000', (REQ, RPOL))
        limb = PolarLimb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

        perp = limb.normal(track)
        self.assertTrue(abs(perp.sep(los) - HALFPI).max() < 1.e-12)

        coords = limb.coords_from_vector3(cept, obs, axes=3)
        self.assertTrue(abs(coords[2]).max() < 1.e6)

        cept2 = limb.vector3_from_coords(coords, obs)
        self.assertTrue((cept2 - cept).norm().median() < 1.e-10)

        ####################

        ground = Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
        limb = PolarLimb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

        perp = limb.normal(track)
        self.assertTrue(abs(perp.sep(los) - HALFPI).max() < 1.e-12)

        coords = limb.coords_from_vector3(cept, obs, axes=3)
        self.assertTrue(abs(coords[2]).max() < 1.e6)

        cept2 = limb.vector3_from_coords(coords, obs)
        self.assertTrue((cept2 - cept).norm().median() < 1.e-10)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
