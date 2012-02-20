################################################################################
# oops_/frame/spin.py: Subclass SpinFrame of class Frame
#
# 2/8/12 Created (MRS)
################################################################################

import numpy as np

from baseclass import Frame
from oops_.array_.all import *
from oops_.transform import Transform

import oops_.registry as registry

class SpinFrame(Frame):
    """Spin is a Frame subclass describing a frame in uniform rotation about
    the Z-axis of another frame.
    """

    def __init__(self, rate, offset, epoch, id, reference):
        """Constructor for a Spin Frame.

        Input:
            rate        the rotation rate of the frame in radians/second.
            offset      the angular offset of the frame at the epoch.
            epoch       the time TDB at which the frame is defined.
            id          the ID under which the frame will be registered.
            reference   the reference frame relative to which this frame spins.
        """

        self.rate = rate
        self.offset = offset
        self.epoch = epoch

        self.frame_id = id

        reference = registry.as_frame(reference)
        self.reference_id = reference.frame_id
        self.origin_id = reference.origin_id
        self.shape = reference.shape

        self.omega = Vector3((0.,0.,self.rate))

        self.register()

########################################

    def transform_at_time(self, time, quick=False):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        time = Scalar.as_scalar(time)
        angle = (time.vals - self.epoch) * self.rate + self.offset

        matrix = np.zeros(time.shape + [3,3])
        matrix[...,2,2] = 1.

        matrix[...,0,0] = np.cos(angle)
        matrix[...,1,1] = matrix[...,0,0]

        matrix[...,0,1] = np.sin(angle)
        matrix[...,1,0] = -matrix[...,0,1]

        return Transform(matrix, self.omega, self.frame_id, self.reference_id)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_SpinFrame(unittest.TestCase):

    def runTest(self):

        # Import here to avoid conflicts
        from oops_.event import Event
        from oops_.transform import Transform

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

        spin1  = SpinFrame(1., 0.,  0., "spin1", "J2000")
        spin2  = SpinFrame(2., 0.,  0., "spin2", "J2000")
        spin3  = SpinFrame(1., 0.,  0., "spin3", "spin2")
        spin1a = SpinFrame(1., 1.,  1., "spin1a", "J2000")

        event = Event(0., (1,0,0), (0,0,0), "SSB", "J2000")
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

        event = Event(0., (1,0,0), (1,2,3), "SSB", "J2000")
        self.assertEqual(event.pos, (1,0,0))
        self.assertEqual(event.vel, (1,2,3))

        event1 = event.wrt_frame("spin1")
        self.assertEqual(event1.pos, (1,0,0))
        self.assertEqual(event1.vel, (1,1,3))

        eps = 1.e-10
        event = Event(eps, (1,0,0), (0,0,0), "SSB", "J2000")

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

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
