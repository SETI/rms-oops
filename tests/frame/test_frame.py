################################################################################
# tests/frame/test_frame.py
################################################################################

import numpy as np
import unittest

import cspyce

from oops.config import QUICK
from oops.body   import Body
from oops.frame  import Frame, QuickFrame, Rotation, SpiceFrame
from oops.path   import SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_Frame(unittest.TestCase):

    def setUp(self):
        Body._undefine_solar_system()
        paths = TEST_SPICE_PREFIX.retrieve(['naif0009.tls', 'pck00010.tpc',
                                            'de421.bsp'])
        for path in paths:
            cspyce.furnsh(path)
        Frame.reset_registry()

    def tearDown(self):
        pass

    def runTest(self):

        # QuickFrame tests

        _ = SpicePath('EARTH', 'SSB')
        _ = SpicePath('MOON', 'SSB')
        _ = SpiceFrame('IAU_EARTH', 'J2000')
        moon  = SpiceFrame('IAU_MOON', 'IAU_EARTH')
        quick = QuickFrame(moon, (-5.,5.),
                        dict(QUICK.dictionary, **{'frame_self_check':3.e-14}))

        # Perfect precision is impossible
        try:
            quick = QuickFrame(moon, (-5.,5.),
                        dict(QUICK.dictionary, **{'frame_self_check':0.}))
            self.assertTrue(False, 'No ValueError raised for PRECISION = 0.')
        except ValueError:
            pass

        # Timing tests...
        test = np.zeros(200000)
        # _ = moon.transform_at_time(test, quick=False)   # takes about 10 sec
        _ = quick.transform_at_time(test)           # takes way less than 1 sec

        Frame.reset_registry()

        ################################
        # Test unregistered frames
        ################################

        j2000 = Frame.as_wayframe('J2000')
        rot_180 = Rotation(np.pi, 2, j2000)
        self.assertTrue(rot_180.frame_id.startswith('TEMPORARY'))

        xform = rot_180.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], -1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], -1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        rot_neg60 = Rotation(-np.pi/3, 2, rot_180)
        self.assertTrue(rot_neg60.frame_id.startswith('TEMPORARY'))

        c60 = 0.5
        s60 = np.sqrt(0.75)

        xform = rot_neg60.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0],  c60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], -s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1],  c60, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        rot_neg120 = Rotation(-np.pi/1.5, 2, rot_neg60)
        self.assertTrue(rot_neg120.frame_id.startswith('TEMPORARY'))

        xform = rot_neg120.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], -c60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], -s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  s60, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], -c60, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        # Attempt to register a frame defined relative to an unregistered frame
        self.assertRaises(ValueError, Rotation, -np.pi, 2, rot_neg60, 'NEG180')

        # Link unregistered frame to registered frame
        identity = rot_neg120.wrt('J2000')

        xform = identity.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], 1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], 1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        # Link registered frame to unregistered frame
        identity = Frame.J2000.wrt(rot_neg120)

        xform = identity.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], 1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0], 0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], 1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

        # Link unregistered frame to registered frame
        identity = rot_neg120.wrt(rot_180)

        xform = identity.transform_at_time(0.)
        self.assertAlmostEqual(xform.matrix.vals[0,0], -1, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[0,1],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,0],  0, delta=1.e-14)
        self.assertAlmostEqual(xform.matrix.vals[1,1], -1, delta=1.e-14)
        self.assertEqual(xform.matrix.vals[2,0], 0)
        self.assertEqual(xform.matrix.vals[2,1], 0)
        self.assertEqual(xform.matrix.vals[0,2], 0)
        self.assertEqual(xform.matrix.vals[1,2], 0)
        self.assertEqual(xform.matrix.vals[2,2], 1)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
