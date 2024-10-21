################################################################################
# tests/path/test_path.py
################################################################################

import numpy as np
import unittest

import cspyce

from oops.config import QUICK
from oops.body   import Body
from oops.frame  import Frame, SpiceFrame
from oops.path   import (Path, LinkedPath, ReversedPath, RelativePath,
                         RotatedPath, QuickPath, LinearPath, SpicePath)
from oops.unittester_support import TEST_SPICE_PFX


class Test_Path(unittest.TestCase):

    def setUp(self):
        cspyce.furnsh(TEST_SPICE_PFX.retrieve('de421.bsp'))
        Path.reset_registry()
        Frame.reset_registry()

    def tearDown(self):
        pass

    def runTest(self):

        Path.USE_QUICKPATHS = False

        self.assertEqual(Path.WAYPOINT_REGISTRY['SSB'], Path.SSB)

        # LinkedPath tests
        _ = SpicePath('SUN', 'SSB')
        earth = SpicePath('EARTH', 'SUN')

        moon = SpicePath('MOON', 'EARTH')
        linked = LinkedPath(moon, earth)

        direct = SpicePath('MOON', 'SUN')

        times = np.arange(-3.e8, 3.01e8, 0.5e7)

        direct_event = direct.event_at_time(times)
        linked_event = linked.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((linked_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((linked_event.vel - direct_event.vel).norm() <= eps).all())

        # RelativePath
        relative = RelativePath(linked, SpicePath('MARS', 'SUN'))
        direct = SpicePath('MOON', 'MARS')

        direct_event = direct.event_at_time(times)
        relative_event = relative.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((relative_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((relative_event.vel - direct_event.vel).norm() <= eps).all())

        # ReversedPath
        reversed = ReversedPath(relative)
        direct = SpicePath('MARS', 'MOON')

        direct_event = direct.event_at_time(times)
        reversed_event = reversed.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((reversed_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((reversed_event.vel - direct_event.vel).norm() <= eps).all())

        # RotatedPath
        rotated = RotatedPath(reversed, SpiceFrame('B1950'))
        direct = SpicePath('MARS', 'MOON', 'B1950')

        direct_event = direct.event_at_time(times)
        rotated_event = rotated.event_at_time(times)

        eps = 1.e-6
        self.assertTrue(((rotated_event.pos - direct_event.pos).norm() <= eps).all())
        self.assertTrue(((rotated_event.vel - direct_event.vel).norm() <= eps).all())

        # QuickPath tests
        moon = SpicePath('MOON', 'EARTH')
        quick = QuickPath(moon, (-5.,5.), QUICK.dictionary)

        # Perfect precision is impossible
        try:
            quick = QuickPath(moon, np.arange(0.,100.,0.0001),
                              dict(QUICK.dictionary, **{'path_self_check':0.}))
            self.assertTrue(False, 'No ValueError raised for PRECISION = 0.')
        except ValueError:
            pass

        # Timing tests...
        test = np.zeros(3000000)
        # _ = moon.event_at_time(test, quick=False)       # takes about 15 sec
        _ = quick.event_at_time(test)                   # takes maybe 2 sec

        Path.reset_registry()
        Frame.reset_registry()

        ################################
        # Test unregistered paths
        ################################

        ssb = Path.as_waypoint('SSB')

        slider1 = LinearPath(([3,0,0],[0,3,0]), 0., ssb)
        self.assertTrue(slider1.path_id.startswith('TEMPORARY'))

        event = slider1.event_at_time(1.)
        self.assertEqual(event.pos, (3,3,0))
        self.assertEqual(event.vel, (0,3,0))

        slider2 = LinearPath(([-2,0,0],[0,0,-2]), 0., slider1)
        self.assertTrue(slider2.path_id.startswith('TEMPORARY'))

        event = slider2.event_at_time(1.)
        self.assertEqual(event.pos, (-2,0,-2))
        self.assertEqual(event.vel, (0,0,-2))

        slider3 = LinearPath(([-1,0,0],[0,-3,2]), 0., slider2)
        self.assertTrue(slider3.path_id.startswith('TEMPORARY'))

        event = slider3.event_at_time(1.)
        self.assertEqual(event.pos, (-1,-3,2))
        self.assertEqual(event.vel, ( 0,-3,2))

        # Link unregistered frame to registered frame
        static = slider3.wrt(ssb)

        event = static.event_at_time(1.)
        self.assertEqual(event.pos, (0,0,0))
        self.assertEqual(event.vel, (0,0,0))

        # Link registered frame to unregistered frame
        static = ssb.wrt(slider3)

        event = static.event_at_time(1.)
        self.assertEqual(event.pos, (0,0,0))
        self.assertEqual(event.vel, (0,0,0))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
