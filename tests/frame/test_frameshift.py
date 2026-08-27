################################################################################
# oops/frame/frameshift.py: Subclass FrameShift of class Frame
################################################################################

import numpy as np
import unittest

import cspyce

from polymath   import Scalar, Vector3
from oops.frame import Frame, FrameShift, SpiceFrame
from oops.path  import Path, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_FrameShift(unittest.TestCase):

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

        np.random.seed(2865)

        DT = 10.
        _ = SpicePath('MARS', 'SSB')
        mars = SpiceFrame('IAU_MARS', 'J2000')

        shifted = FrameShift(DT, mars, frame_id='+')
        self.assertEqual(shifted.frame_id, 'IAU_MARS_SHIFT')
        self.assertEqual(shifted.dt, DT)
        self.assertIsNone(shifted.link)
        self.assertEqual(shifted.reference, mars.reference)

        # The shifted frame at time t matches the original frame at time t + dt
        time = Scalar(1.e8 + np.arange(10) * 1000.)
        self.assertEqual(shifted.transform_at_time(time).matrix,
                         mars.transform_at_time(time + DT).matrix)
        self.assertEqual(shifted.transform_at_time(time).omega,
                         mars.transform_at_time(time + DT).omega)

        # Mars turns steadily about its pole, so the angle a shift introduces grows in
        # proportion to the shift
        t0 = Scalar(1.e8)
        v0 = mars.transform_at_time(t0).rotate(Vector3.XAXIS)
        sep1 = shifted.transform_at_time(t0).rotate(Vector3.XAXIS).sep(v0)
        shifted2 = FrameShift(2. * DT, mars, frame_id='mars_shifted_x2')
        sep2 = shifted2.transform_at_time(t0).rotate(Vector3.XAXIS).sep(v0)
        self.assertGreater(sep1.vals, 0.)
        self.assertAlmostEqual((sep2 / sep1).vals, 2., places=6)

        # A zero shift leaves the frame alone
        unshifted = FrameShift(0., mars, frame_id='mars_unshifted')
        self.assertEqual(unshifted.transform_at_time(time).matrix,
                         mars.transform_at_time(time).matrix)

        # A linked FrameShift tracks the offset of the object it is linked to
        linked = FrameShift(shifted, mars, frame_id='mars_shifted_2')
        self.assertIs(linked.link, shifted)
        self.assertEqual(linked.dt, DT)
        self.assertEqual(linked.params, (DT,))
        self.assertEqual(linked.nparams, 1)

        shifted.set_params(np.array([2. * DT]))
        linked.refresh()
        self.assertEqual(linked.dt, 2. * DT)
        self.assertEqual(linked.transform_at_time(time).matrix,
                         mars.transform_at_time(time + 2. * DT).matrix)

        # Freezing severs the link but preserves the offset
        frozen = FrameShift(linked, mars, frame_id='mars_shifted_3', freeze=True)
        self.assertTrue(frozen.is_frozen)
        self.assertIsNone(frozen.link)
        self.assertEqual(frozen.dt, 2. * DT)

        # Freezing an object also freezes the Fittable objects it was built from, so
        # the source can no longer be refit
        self.assertTrue(linked.is_frozen)
        self.assertTrue(shifted.is_frozen)
        self.assertRaisesRegex(ValueError, 'frozen',
                               shifted.set_params, np.array([3. * DT]))

################################################################################
