################################################################################
# oops/path/quickpath.py: Subclass QuickPath of class Path
################################################################################

import numpy as np
import unittest

import cspyce

from polymath    import Scalar, Vector3
from oops.config import QUICK
from oops.frame  import Frame
from oops.path   import FixedPath, Path, QuickPath, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_QuickPath(unittest.TestCase):

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

        np.random.seed(9033)

        mars = SpicePath('MARS', 'SSB')
        epoch = 1.e8
        time = Scalar(epoch + np.arange(0., 100., 0.01))

        ########################################
        # Tabulating a Path does not spawn a second, nested QuickPath
        ########################################

        # SpicePath quickens itself when handed an array of times, so the tabulation
        # inside QuickPath must not re-enter that machinery. The span is short enough
        # that a QuickPath of the tabulation times would otherwise be judged worthwhile.
        short_time = Scalar(epoch + np.arange(0., 0.5, 0.001))
        self.assertIsInstance(mars.quick_path(short_time, quick={}), QuickPath)
        self.assertEqual(len(mars._quickpaths), 1)

        ########################################
        # A Path whose state is fixed in time is never tabulated
        ########################################

        # FixedPath returns one position and velocity regardless of the times requested,
        # so a QuickPath could not interpolate it and would gain nothing if it could
        fixed = FixedPath(Vector3((1.e3, 0., 0.)), mars, path_id='fixed_path')
        self.assertFalse(fixed._USE_QUICKPATHS)
        self.assertIs(fixed.quick_path(time, quick={}), fixed)

        self.assertRaises(ValueError, QuickPath, fixed, epoch, epoch + 100.,
                          QUICK.dictionary)

        ########################################
        # A composite Path inherits the opt-in of the paths it combines
        ########################################

        # The fixed offset above contributes nothing, but the SpicePath it is measured
        # from does, so the composite is worth tabulating
        linked = fixed.wrt(Path.SSB, Frame.J2000)
        self.assertTrue(linked._USE_QUICKPATHS)
        quick = linked.quick_path(time, quick={})
        self.assertIsInstance(quick, QuickPath)

        exact = linked.event_at_time(time, quick=False)
        interpolated = quick.event_at_time(time, quick=False)
        self.assertLess(np.max(np.abs((interpolated.pos - exact.pos).norm().vals)), 1.e-6)

################################################################################
