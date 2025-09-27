##########################################################################################
# oops/frame/inclinedframe.py: Subclass InclinedFrame of class Frame
##########################################################################################

from polymath             import Qube, Scalar
from oops.fittable        import Fittable_
from oops.frame           import Frame
from oops.frame.rotation  import Rotation
from oops.frame.spinframe import SpinFrame


class InclinedFrame(Frame):
    """InclinedFrame is a Frame subclass describing a frame that is inclined to the
    equator of another frame.

    It is defined by an inclination, a node at epoch, and a nodal regression rate. This
    frame is oriented to be "nearly inertial," meaning that a longitude in the new frame
    is determined by measuring from the reference longitude in the reference frame, along
    that frame's equator to the ascending node, and thence along the ascending node.
    """

    _FRAME_IDS = {}

    def __init__(self, inc, node, rate, epoch, reference, *, despin=True, frame_id=None):
        """Constructor for a InclinedFrame.

        Parameters:
            inc (Scalar, array-like, or float): Inclination angle in radians.
            node (Scalar, array-like, or float): Longitude of None at epoch in radians.
            rate (Scalar, array-like, or float): Rate of nodal presession in radians/s.
            epoch Scalar, array-like, or float): Time in seconds TDB at which the `node`
                applies.
            reference (Frame or str): Frame or Frame ID describing the central planet of
                the inclined plane.
            despin (bool, optional): True for a nearly inertial frame, in which the x and
                yaxes vary as little as possible while the zaxis rotates; False for a
                frame in which the x axis is tied to the ascending node.
            frame_id (str, optional): The ID under which the frame will be registered;
                None to leave the frame unregistered

        Note that inc, node, rate and epoch can all be Scalars of arbitrary shape. The
        shape of the InclinedFrame is the result of broadcasting all these shapes
        together.
        """

        self.inc = Scalar.as_scalar(inc)
        self.node = Scalar.as_scalar(node)
        self.rate = Scalar.as_scalar(rate)
        self.epoch = Scalar.as_scalar(epoch)
        self.despin = bool(despin)

        # Required attributes
        self.reference = Frame.as_wayframe(reference)
        self.origin = self.reference.origin
        self.shape = Qube.broadcast(self.inc, self.node, self.rate, self.epoch)
        self.frame_id = self._recover_id(frame_id)

        # Update wayframe and frame_id; register if not temporary
        self.register()
        self._refresh()
        self._cache_id()

    def _refresh(self):
        self.spin1 = SpinFrame(self.node, self.rate, self.epoch, axis=2,
                               reference=self.reference)
        self.rotate = Rotation(self.inc, axis=0, reference=self.spin1)
        self.rotate.freeze()

        if self.despin:
            self.spin2 = SpinFrame(-self.node, -self.rate, self.epoch, axis=2,
                                   reference=self.rotate)
        else:
            self.spin2 = None

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.inc, self.node, self.rate, self.epoch, self.reference, self.despin)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.inc, self.node, self.rate, self.epoch,
                Frame.as_primary_frame(self.reference), self.despin, self._state_id())

    def __setstate__(self, state):
        (inc, node, rate, epoch, reference, despin, frame_id) = state
        self.__init__(inc, node, rate, epoch, reference=reference, despin=despin,
                      frame_id=frame_id)
        Fittable_.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, quick=False):
        """Transform that rotates coordinates from the reference frame to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.
        """

        xform = self.spin1.transform_at_time(time)
        xform = self.rotate.transform_at_time(time).rotate_transform(xform)

        if self.spin2:
            xform = self.spin2.transform_at_time(time).rotate_transform(xform)

        return xform

    def node_at_time(self, time):
        """The vector defining the ascending node of this frame's XY plane relative to
        the XY frame of its reference.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Vector3): The unit vector pointing in the direction of the ascending node.
        """

        # Locate the ascending nodes in the reference frame
        time = Scalar.as_scalar(time)
        return (self.node + self.rate * (time - self.epoch)) % Scalar.TWOPI

##########################################################################################
