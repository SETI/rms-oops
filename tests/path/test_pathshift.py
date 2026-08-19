################################################################################
# oops/path/pathshift.py: Subclass PathShift of class Path
################################################################################

import numpy as np
import unittest

import cspyce

from polymath   import Scalar
from oops.frame import Frame
from oops.path  import Path, PathShift, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_PathShift(unittest.TestCase):

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

        DT = 10.
        mars = SpicePath('MARS', 'SSB')
        shifted = PathShift(DT, mars, path_id='mars_shifted')
        self.assertEqual(shifted.dt, DT)

        # The shifted path at time t matches the original path at time t + dt
        time = Scalar(1.e8 + np.arange(10) * 1000.)
        self.assertEqual(shifted.event_at_time(time).pos,
                         mars.event_at_time(time + DT).pos)
        self.assertEqual(shifted.event_at_time(time).vel,
                         mars.event_at_time(time + DT).vel)

        # A linked PathShift tracks the offset of the object it is linked to
        linked = PathShift(shifted, mars, path_id='mars_shifted_2')
        self.assertEqual(linked.dt, DT)

        shifted.set_params(np.array([2. * DT]))
        linked.refresh()
        self.assertEqual(linked.dt, 2. * DT)
        self.assertEqual(linked.event_at_time(time).pos,
                         mars.event_at_time(time + 2. * DT).pos)

        # Freezing severs the link but preserves the offset
        frozen = PathShift(linked, mars, path_id='mars_shifted_3', freeze=True)
        self.assertTrue(frozen.is_frozen)
        self.assertEqual(frozen.dt, 2. * DT)

################################################################################
