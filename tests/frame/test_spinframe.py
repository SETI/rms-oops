################################################################################
# tests/frame/test_spinframe.py
################################################################################

import numpy as np
import unittest

from polymath   import Scalar, Vector3
from oops       import Path
from oops.frame import Frame, SpinFrame


class Test_SpinFrame(unittest.TestCase):

    def runTest(self):

        np.random.seed(6521)

        # Import here to avoid conflicts
        from oops.event import Event

        Frame.reset_registry()
        Path.reset_registry()

        spin1 = SpinFrame(0., 1., 0., 2, "J2000", "spin1")
        _ = SpinFrame(0., 2., 0., 2, "J2000", "spin2")
        _ = SpinFrame(0., 1., 0., 2, "spin2", "spin3")
        _ = SpinFrame(1., 1., 1., 2, "J2000", "spin1a")

        event = Event(Scalar.ZERO, Vector3.XAXIS, "SSB", "J2000")
        self.assertEqual(event.pos, (1,0,0))
        self.assertEqual(event.vel, (0,0,0))

        event1 = event.wrt_frame("spin1")
        self.assertEqual(event1.pos, (1, 0,0))
        self.assertEqual(event1.vel, (0,-1,0))

        self.assertEqual(event.pos, (1,0,0))
        self.assertEqual(event.vel, (0,0,0))

        event2 = event.wrt_frame("spin2")
        self.assertEqual(event2.pos, (1, 0,0))
        self.assertEqual(event2.vel, (0,-2,0))

        event3 = event.wrt_frame("spin3")
        self.assertEqual(event3.pos, (1, 0,0))
        self.assertEqual(event3.vel, (0,-3,0))

        event = Event(Scalar.ZERO, (Vector3.XAXIS,(1,2,3)), "SSB", "J2000")
        self.assertEqual(event.pos, (1,0,0))
        self.assertEqual(event.vel, (1,2,3))

        event1 = event.wrt_frame("spin1")
        self.assertEqual(event1.pos, (1,0,0))
        self.assertEqual(event1.vel, (1,1,3))

        eps = 1.e-10
        event = Event(eps, Vector3.XAXIS, "SSB", "J2000")

        event1 = event.wrt_frame("spin1")
        self.assertEqual(event1.pos, (1, -eps,0))
        self.assertEqual(event1.vel, (-eps,-1,0))

        event2 = event.wrt_frame("spin2")
        self.assertEqual(event2.pos, (1, -2*eps,0))
        self.assertEqual(event2.vel, (-4*eps,-2,0))

        event3 = event.wrt_frame("spin3")
        self.assertEqual(event3.pos, (1, -3*eps,0))
        self.assertEqual(event3.vel, (-9*eps,-3,0))

        event1a = event.wrt_frame("spin1a")
        self.assertTrue((event1a.pos - (1, -eps,0)).norm() < 1.e-15)
        self.assertTrue((event1a.vel - (-eps,-1,0)).norm() < 1.e-15)

        # Test time-derivatives of transforms
        time = Scalar(np.random.randn(400))
        pos  = Vector3(np.random.randn(400,3))
        vel  = Vector3(np.random.randn(400,3))

        dt = 1.e-6
        tr0 = spin1.transform_at_time(time)
        tr1 = spin1.transform_at_time(time + dt)

        (pos0, vel0) = tr0.rotate_pos_vel(pos, vel)
        (pos1, vel1) = tr1.rotate_pos_vel(pos + vel*dt, vel)
        dpos_dt_test = (pos1 - pos0) / dt
        self.assertTrue(abs(dpos_dt_test - vel0).max() < 1.e-5)

        (pos0, vel0) = tr0.unrotate_pos_vel(pos, vel)
        (pos1, vel1) = tr1.unrotate_pos_vel(pos + vel*dt, vel)
        dpos_dt_test = (pos1 - pos0) / dt
        self.assertTrue(abs(dpos_dt_test - vel0).max() < 1.e-5)

        pos0 = tr0.rotate(pos, derivs=True)
        pos1 = tr1.rotate(pos, derivs=False)
        dpos_dt_test = (pos1 - pos0) / dt
        self.assertTrue(abs(dpos_dt_test - pos0.d_dt).max() < 1.e-5)

        pos0 = tr0.unrotate(pos, derivs=True)
        pos1 = tr1.unrotate(pos, derivs=False)
        dpos_dt_test = (pos1 - pos0) / dt
        self.assertTrue(abs(dpos_dt_test - pos0.d_dt).max() < 1.e-5)

        Frame.reset_registry()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
