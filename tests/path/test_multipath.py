################################################################################
# tests/path/test_multipath.py
################################################################################

import os
import unittest

import cspyce

from oops      import Frame
from oops.path import Path, MultiPath, SpicePath
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


class Test_MultiPath(unittest.TestCase):

    def runTest(self):

        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "naif0009.tls"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "pck00010.tpc"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "de421.bsp"))

        Path.reset_registry()
        Frame.reset_registry()

        sun   = SpicePath("SUN", "SSB")
        earth = SpicePath("EARTH", "SSB")
        moon  = SpicePath("MOON", "EARTH")

        test = MultiPath([sun,earth,moon], "SSB")

        self.assertEqual(test.path_id, "SUN+others")
        self.assertEqual(test.shape, (3,))

        # Single time
        event0 = test.event_at_time(0.)
        self.assertEqual(event0.shape, (3,))

        # Triple of times, shape = [3]
        event012 = test.event_at_time((0., 1.e5, 2.e5))
        self.assertEqual(event012.shape, (3,))

        self.assertTrue(event012.pos[0] == event0.pos[0])
        self.assertTrue(event012.vel[0] == event0.vel[0])
        self.assertTrue(event012.pos[1] != event0.pos[1])
        self.assertTrue(event012.vel[1] != event0.vel[1])
        self.assertTrue(event012.pos[2] != event0.pos[2])
        self.assertTrue(event012.vel[2] != event0.vel[2])

        # Times shaped [2,1]
        event01x = test.event_at_time([[0.], [1.e5]])
        self.assertEqual(event01x.shape, (2,3))

        self.assertTrue(event01x.pos[0,0] == event0.pos[0])
        self.assertTrue(event01x.vel[0,0] == event0.vel[0])
        self.assertTrue(event01x.pos[0,1] == event0.pos[1])
        self.assertTrue(event01x.vel[0,1] == event0.vel[1])
        self.assertTrue(event01x.pos[0,2] == event0.pos[2])
        self.assertTrue(event01x.vel[0,2] == event0.vel[2])

        self.assertTrue(event01x.pos[1,1] == event012.pos[1])
        self.assertTrue(event01x.vel[1,1] == event012.vel[1])
        self.assertTrue(event01x.pos[1,2] != event012.pos[2])
        self.assertTrue(event01x.vel[1,2] != event012.pos[2])

        # Triple of times, at all times, shape [3,1]
        event012a = test.event_at_time([[0.], [1.e5], [2.e5]])
        self.assertEqual(event012a.shape, (3,3))

        self.assertTrue(event012a.pos[0,:] == event0.pos)
        self.assertTrue(event012a.vel[0,:] == event0.vel)

        self.assertTrue(event012a.pos[0,0] == event012.pos[0])
        self.assertTrue(event012a.vel[0,0] == event012.vel[0])
        self.assertTrue(event012a.pos[1,1] == event012.pos[1])
        self.assertTrue(event012a.vel[1,1] == event012.vel[1])
        self.assertTrue(event012a.pos[2,2] == event012.pos[2])
        self.assertTrue(event012a.vel[2,2] == event012.vel[2])

        self.assertTrue(event012a.pos[0:2] == event01x.pos)
        self.assertTrue(event012a.vel[0:2] == event01x.vel)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
