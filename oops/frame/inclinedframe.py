################################################################################
# oops/frame/inclinedframe.py: Subclass InclinedFrame of class Frame
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from .           import Frame
from .spinframe  import SpinFrame
from .rotation   import Rotation
from ..transform import Transform
from ..path      import Path
import oops.constants as constants

class InclinedFrame(Frame):
    """InclinedFrame is a Frame subclass describing a frame that is inclined to
    the equator of another frame. It is defined by an inclination, a node at
    epoch, and a nodal regression rate. This frame is oriented to be "nearly
    inertial," meaning that a longitude in the new frame is determined by
    measuring from the reference longitude in the reference frame, along that
    frame's equator to the ascending node, and thence along the ascending node.
    """

    PACKRAT_ARGS = ['inc', 'node', 'rate', 'epoch', 'reference', 'despin',
                    'frame_id']

    #===========================================================================
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
                        to leave the frame unregistered.

        Note that inc, node, rate and epoch can all be scalars of arbitrary
        shape. The shape of the InclinedFrame is the result of broadcasting all
        these shapes together.
        """

        self.inc = Scalar.as_scalar(inc)
        self.node = Scalar.as_scalar(node)
        self.rate = Scalar.as_scalar(rate)
        self.epoch = Scalar.as_scalar(epoch)

        self.shape = Qube.broadcast(self.inc, self.node, self.rate, self.epoch)

        self.frame_id  = id
        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.keys      = set()

        self.spin1  = SpinFrame(self.node, self.rate, self.epoch, axis=2,
                                reference=self.reference)
        self.rotate = Rotation(self.inc, axis=0, reference=self.spin1)

        self.despin = despin
        if despin:
            self.spin2 = SpinFrame(-self.node, -self.rate, self.epoch, axis=2,
                                   reference=self.rotate)
        else:
            self.spin2 = None

        # Update wayframe and frame_id; register if not temporary
        self.register()

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """The Transform into the this Frame at a Scalar of times."""

        xform = self.spin1.transform_at_time(time)
        xform = self.rotate.transform_at_time(time).rotate_transform(xform)

        if self.spin2:
            xform = self.spin2.transform_at_time(time).rotate_transform(xform)

        return xform

    #===========================================================================
    def node_at_time(self, time):
        """The longitude of ascending node at the specified time."""

        # Locate the ascending nodes in the reference frame
        return (self.node + self.rate * (Scalar.as_scalar(time)
                                         - self.epoch)) % constants.TWOPI

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
