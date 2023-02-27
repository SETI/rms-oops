################################################################################
# oops/frame/inclinedframe.py: Subclass InclinedFrame of class Frame
################################################################################

from polymath import Qube, Scalar

from .                 import Frame
from .rotation         import Rotation
from .spinframe        import SpinFrame
from ..frame.poleframe import PoleFrame
import oops.constants as constants

class InclinedFrame(Frame):
    """InclinedFrame is a Frame subclass describing a frame that is inclined to
    the equator of another frame.

    It is defined by an inclination, a node at epoch, and a nodal regression
    rate. This frame is oriented to be "nearly inertial," meaning that a
    longitude in the new frame is determined by measuring from the reference
    longitude in the reference frame, along that frame's equator to the
    ascending node, and thence along the ascending node.
    """

    FRAME_IDS = {}  # frame_id to use if a frame already exists upon un-pickling

    #===========================================================================
    def __init__(self, inc, node, rate, epoch, reference, despin=True,
                       frame_id=None, unpickled=False):
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

            frame_id    the ID under which the frame will be registered; None
                        to leave the frame unregistered.

            unpickled   True if this frame has been read from a pickle file.

        Note that inc, node, rate and epoch can all be scalars of arbitrary
        shape. The shape of the InclinedFrame is the result of broadcasting all
        these shapes together.
        """

        self.inc = Scalar.as_scalar(inc)
        self.node = Scalar.as_scalar(node)
        self.rate = Scalar.as_scalar(rate)
        self.epoch = Scalar.as_scalar(epoch)

        self.shape = Qube.broadcast(self.inc, self.node, self.rate, self.epoch)

        self.frame_id  = frame_id
        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.keys      = set()

        self.spin1  = SpinFrame(self.node, self.rate, self.epoch, axis=2,
                                reference=self.reference)
        self.rotate = Rotation(self.inc, axis=0, reference=self.spin1)

        self.despin = bool(despin)
        if despin:
            self.spin2 = SpinFrame(-self.node, -self.rate, self.epoch, axis=2,
                                   reference=self.rotate)
        else:
            self.spin2 = None

        # Update wayframe and frame_id; register if not temporary
        self.register(unpickled=unpickled)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.frame_id in Frame.WAYFRAME_REGISTRY):
                key = (self.inc.vals, self.node.vals, self.rate.vals,
                       self.epoch.vals, self.reference.frame_id, self.despin)
                InclinedFrame.FRAME_IDS[key] = self.frame_id

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.inc, self.node, self.rate, self.epoch, self.reference,
                self.despin, self.shape)

    def __setstate__(self, state):
        # If this frame matches a pre-existing frame, re-use its ID
        (inc, node, rate, epoch, reference, despin, shape) = state
        if shape == ():
            key = (inc.vals, node.vals, rate.vals, epoch.vals,
                   reference.frame_id, despin)
            frame_id = PoleFrame.FRAME_IDS.get(key, None)
        else:
            frame_id = None

        self.__init__(inc, node, rate, epoch, reference, despin,
                      frame_id=frame_id, unpickled=True)

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
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
