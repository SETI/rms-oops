################################################################################
# oops/frame/quickframe.py: Subclass QuickFrame of class Frame
################################################################################

import numpy as np
import unittest

import cspyce

from polymath   import Matrix3, Quaternion, Scalar, Vector3
from oops.frame import (Frame, Cmatrix, PosTargFrame, QuickFrame, Rotation,
                        SpiceFrame, SpinFrame, TwoVectorFrame)
from oops.path  import Path, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_QuickFrame(unittest.TestCase):

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

        np.random.seed(4417)

        _ = SpicePath('MARS', 'SSB')
        mars = SpiceFrame('IAU_MARS', 'J2000')

        epoch = 1.e8
        time = Scalar(epoch + np.arange(0., 100., 0.01))

        ########################################
        # Tabulating a Frame does not spawn a second, nested QuickFrame
        ########################################

        # SpiceFrame quickens itself when handed an array of times, so the tabulation
        # inside QuickFrame must not re-enter that machinery. The span is short enough
        # that a QuickFrame of the tabulation times would otherwise be judged worthwhile.
        short_time = Scalar(epoch + np.arange(0., 0.5, 0.001))
        self.assertIsInstance(mars.quick_frame(short_time, quick={}), QuickFrame)
        self.assertEqual(len(mars._quickframes), 1)

        # The same must hold when the frame being tabulated is a composite, which
        # requires LinkedFrame to forward `quick` to the frames it combines. The times
        # are well clear of the tabulation above, so a second QuickFrame of IAU_MARS
        # could not be mistaken for a re-use of the first.
        linked_time = short_time + 1000.
        twovector = TwoVectorFrame(mars, Vector3.XAXIS, 'X', Vector3.YAXIS, 'Y',
                                   frame_id='nested_twovector')
        linked = twovector.wrt(Frame.J2000)
        self.assertIsInstance(linked.quick_frame(linked_time, quick={}), QuickFrame)
        self.assertEqual(len(mars._quickframes), 1)

        ########################################
        # A Frame whose transform is fixed in time is never tabulated
        ########################################

        # These Frames return one Transform regardless of the times requested, so a
        # QuickFrame could not interpolate them and would gain nothing if it could
        fixed = [Cmatrix(Matrix3.IDENTITY, reference=mars, frame_id='fixed_cmatrix'),
                 PosTargFrame(1.e-5, 2.e-5, mars, frame_id='fixed_postarg'),
                 Rotation(0.3, 2, mars, frame_id='fixed_rotation'),
                 TwoVectorFrame(mars, Vector3.XAXIS, 'X', Vector3.YAXIS, 'Y',
                                frame_id='fixed_twovector')]

        for frame in fixed:
            self.assertFalse(frame._USE_QUICKFRAMES)
            self.assertIs(frame.quick_frame(time, quick={}), frame)

            # ...but the composite with a time-dependent reference still is tabulated
            linked = frame.wrt(Frame.J2000)
            self.assertTrue(linked._USE_QUICKFRAMES)
            self.assertIsInstance(linked.quick_frame(time, quick={}), QuickFrame)

        ########################################
        # Tabulating a fixed Frame raises a meaningful error
        ########################################

        cmatrix = fixed[0]
        self.assertRaises(ValueError, QuickFrame, cmatrix, epoch, epoch + 100.)

        ########################################
        # Quaternions q and -q describe the same rotation, so the tabulated values can
        # reverse sign where the rotation angle passes pi; the splines require them to
        # be continuous
        ########################################

        # 110 seconds at 0.1 rad/s sweeps through pi more than three times
        spin = SpinFrame(0., 0.1, epoch, 2, mars, frame_id='fast_spin')
        spin_wrt_j2000 = spin.wrt(Frame.J2000)

        quick = spin_wrt_j2000.quick_frame(time, quick={})
        self.assertIsInstance(quick, QuickFrame)

        exact = spin_wrt_j2000.transform_at_time(time, quick=False)
        interpolated = quick.transform_at_time(time)
        error = np.max(np.abs(interpolated.matrix.vals - exact.matrix.vals))
        self.assertLess(error, 1.e-8)

        # The tabulation itself must be free of sign reversals
        quats = Quaternion.as_quaternion(quick._xforms.matrix).vals
        unwrapped = QuickFrame._unwrap_quaternions(quats)
        self.assertTrue(np.any(np.sum(quats[:-1] * quats[1:], axis=-1) < 0.))
        self.assertTrue(np.all(np.sum(unwrapped[:-1] * unwrapped[1:], axis=-1) > 0.))

        # Unwrapping preserves the rotations it describes
        before = Matrix3.as_matrix3(Quaternion(quats)).vals
        after = Matrix3.as_matrix3(Quaternion(unwrapped)).vals
        self.assertLess(np.max(np.abs(after - before)), 1.e-14)

        # A tabulation without sign reversals is returned unchanged
        self.assertIs(QuickFrame._unwrap_quaternions(unwrapped), unwrapped)

################################################################################
