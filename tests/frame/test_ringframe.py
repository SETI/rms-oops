################################################################################
# oops/frame/ringframe.py: Subclass RingFrame of class Frame
################################################################################

import numpy as np
import unittest

import cspyce

from polymath   import Scalar, Vector3
from oops       import Event
from oops.body  import Body
from oops.frame import Frame, RingFrame, SpiceFrame
from oops.path  import Path, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_RingFrame(unittest.TestCase):

    def setUp(self):
        paths = TEST_SPICE_PREFIX.retrieve(["naif0009.tls", "pck00010.tpc",
                                            "de421.bsp"])
        for path in paths:
            cspyce.furnsh(path)
        Path.reset_registry()
        Frame.reset_registry()

    def tearDown(self):
        pass

    def runTest(self):

        # Imports are here to reduce conflicts

        np.random.seed(2492)

        _ = SpicePath("MARS", "SSB")
        planet = SpiceFrame("IAU_MARS", "J2000")
        rings  = RingFrame(planet)
        self.assertEqual(Frame.as_wayframe("IAU_MARS"), planet.wayframe)
        self.assertEqual(Frame.as_wayframe("IAU_MARS_DESPUN"), rings.wayframe)

        time = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6)
        event = Event(time, (posvel[...,0:3], posvel[...,3:6]), "SSB", "J2000")
        rotated = event.wrt_frame("IAU_MARS")
        fixed   = event.wrt_frame("IAU_MARS_DESPUN")

        # Confirm Z axis is tied to planet's pole
        diff = Scalar(rotated.pos.mvals[...,2]) - Scalar(fixed.pos.mvals[...,2])
        self.assertTrue(np.all(np.abs(diff.values < 1.e-14)))

        # Confirm X-axis is always in the J2000 equator
        xaxis = Event(time, Vector3.XAXIS, "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(np.abs(test.pos.mvals[...,2] < 1.e-14)))

        # Confirm it's at the ascending node
        xaxis = Event(time, (1,1.e-13,0), "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(test.pos.mvals[...,1] > 0.))

        # Check that pole wanders when epoch is fixed
        rings2 = RingFrame(planet, 0.)
        self.assertEqual(Frame.as_wayframe("IAU_MARS_INERTIAL"), rings2.wayframe)
        inertial = event.wrt_frame("IAU_MARS_INERTIAL")

        diff = Scalar(rotated.pos.mvals[...,2]) - Scalar(inertial.pos.mvals[...,2])
        self.assertTrue(np.all(np.abs(diff.values) < 1.e-4))
        self.assertTrue(np.mean(np.abs(diff.values) > 1.e-8))

#         Path.reset_registry()
#         Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
