################################################################################
# oops_/frame/inclinedframe.py: Subclass InclinedFrame of class Frame
#
# 3/17/12 MRS - created.
################################################################################

import numpy as np

from oops_.frame.frame_ import Frame
from oops_.frame.spinframe import SpinFrame
from oops_.frame.rotation import Rotation
from oops_.array.all import *
from oops_.config import QUICK
from oops_.transform import Transform

import oops_.registry as registry

class InclinedFrame(Frame):
    """InclinedFrame is a Frame subclass describing a frame that is inclined to
    the equator of another frame. It is defined by an inclination, a node at
    epoch, and a nodal regression rate. This frame is oriented to be "nearly
    inertial," meaning that a longitude in the new frame is determined by
    measuring from the reference longitude in the reference frame, along that
    frame's equator to the ascending node, and thence along the ascending node.
    """

    def __init__(self, inc, node, rate, epoch, reference, despin=True, id=None):
        """Constructor for a InclinedFrame.

        Input:
            inc         the inclination of the plane in radians.

            node        the longitude of ascending node of the inclined plane
                        at the specified epoch, in radians. This measured
                        relative to the ascending node of the planet's equator
                        relative to its parent frame, which is typically J2000.

            rate        the nodal regression rate of the inclined plane in
                        radians per second. Should be negative for a ring about
                        an oblate planet.

            epoch       the time TDB at which the node is defined.

            reference   a reference frame describing the central planet of the
                        inclined plane.

            despin      True to return a nearly inertial frame; False to return
                        a frame in which the x-axis is tied to the ascending
                        node.

            id          the ID under which the frame will be registered; None
                        to use a temporary ID.

        Note that inc, node, rate and epoch can all be scalars of arbitrary
        shape. The shape of the InclinedFrame is the result of broadcasting all
        these shapes together.
        """

        self.inc = inc
        self.node = node
        self.rate = rate
        self.epoch = epoch
        self.despin = despin

        self.shape = Array.broadcast_shape((inc, node, rate, epoch))

        self.frame_id = id
        reference = registry.as_frame(reference)
        self.reference_id = reference.frame_id
        self.origin_id = reference.origin_id

        self.spin1  = SpinFrame(self.node, self.rate, self.epoch, axis=2,
                                reference=self.reference_id)
        self.rotate = Rotation(self.inc, axis=0,
                                reference=self.spin1.frame_id)

        if self.despin:
            self.spin2 = SpinFrame(-self.node, -self.rate, self.epoch, axis=2,
                                   reference=self.rotate.frame_id)

        self.register()

########################################

    def transform_at_time(self, time, quick=QUICK):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        xform = self.spin1.transform_at_time(time)
        xform = self.rotate.transform_at_time(time).rotate_transform(xform)

        if self.despin:
            xform = self.spin2.transform_at_time(time).rotate_transform(xform)

        return xform

########################################

    def node_at_time(self, time):
        """Returns the longitude of ascending node at the specified time."""

        # Locate the ascending nodes in the reference frame
        return (self.node + rate * (time - self.epoch)) & (2*np.pi)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_InclinedFrame(unittest.TestCase):

    def runTest(self):

        # Note: Unit testing is performed in surface/orbitplane.py
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
