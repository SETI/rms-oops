################################################################################
# oops/frame/navigation.py: Subclass Navigation of class Frame
################################################################################

import numpy as np
import unittest

import cspyce

from polymath   import Scalar, Vector3
from oops.frame import Frame, Navigation, SpiceFrame
from oops.path  import Path, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_Navigation(unittest.TestCase):

    def setUp(self):
        paths = TEST_SPICE_PREFIX.retrieve(['naif0009.tls',
                                            'pck00010.tpc',
                                            'de421.bsp'])
        for path in paths:
            cspyce.furnsh(path)
        Path._reset_caches()
        Frame._reset_caches()

    def tearDown(self):
        Path._reset_caches()
        Frame._reset_caches()

    def runTest(self):

        _ = SpicePath('MARS', 'SSB')
        mars = SpiceFrame('IAU_MARS', 'J2000')
        time = Scalar(1.e8)

        # Two small rotations about perpendicular axes tilt the Z-axis by their
        # root sum of squares
        (ay, ax) = (1.e-3, 2.e-3)
        nav = Navigation((ay, ax), mars, frame_id='+')
        self.assertEqual(nav.frame_id, 'IAU_MARS_NAV')
        self.assertEqual(nav.nparams, 2)
        self.assertIsInstance(type(nav).link, property)
        self.assertIsNone(nav.link)

        zaxis = nav.transform_at_time(time).rotate(Vector3.ZAXIS)
        self.assertAlmostEqual(zaxis.sep(Vector3.ZAXIS).vals,
                               np.sqrt(ay**2 + ax**2), places=8)

        # A third angle rotates about the Z-axis, leaving the Z-axis untouched
        nav3 = Navigation((0., 0., 0.1), mars, frame_id='nav3')
        self.assertEqual(nav3.nparams, 3)
        self.assertEqual(nav3.transform_at_time(time).rotate(Vector3.ZAXIS),
                         Vector3.ZAXIS)
        xaxis = nav3.transform_at_time(time).rotate(Vector3.XAXIS)
        self.assertAlmostEqual(xaxis.sep(Vector3.XAXIS).vals, 0.1, places=12)

        # A linked Navigation tracks the angles of the object it is linked to
        linked = Navigation(nav, mars, frame_id='nav_linked')
        self.assertIs(linked.link, nav)
        self.assertEqual(linked.nparams, 2)

        linked.set_params(np.array([5.e-3, 6.e-3]))
        nav.refresh()
        self.assertEqual(tuple(nav.angles), (5.e-3, 6.e-3))
        self.assertEqual(tuple(linked.angles), (5.e-3, 6.e-3))

        # Freezing severs the link but preserves the angles
        frozen = Navigation(linked, mars, frame_id='nav_frozen', freeze=True)
        self.assertTrue(frozen.is_frozen)
        self.assertEqual(tuple(frozen.angles), (5.e-3, 6.e-3))

################################################################################
