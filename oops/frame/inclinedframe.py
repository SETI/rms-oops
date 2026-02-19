##########################################################################################
# oops/frame/inclinedframe.py: Subclass InclinedFrame of class Frame
##########################################################################################

from polymath             import Qube, Scalar
from oops.frame           import Frame
from oops.frame.rotation  import Rotation
from oops.frame.spinframe import SpinFrame
import oops.mutable as mutable


class InclinedFrame(Frame):
    """InclinedFrame is a Frame subclass describing a frame that is inclined to the
    equator of another frame.

    It is defined by an inclination, a node at epoch, and a nodal regression rate. This
    frame is oriented to be "nearly inertial," meaning that a longitude in the new frame
    is determined by measuring from the reference longitude in the reference frame, along
    that frame's equator to the ascending node, and thence along the ascending node.
    """

    _WAYFRAMES = {}

    def __init__(self, inc, node, rate, epoch, reference, *, despin=True, frame_id=None):
        """Constructor for a InclinedFrame.

        Parameters:
            inc (Scalar, array-like, or float): Inclination angle in radians.
            node (Scalar, array-like, or float): Longitude of None at epoch in radians.
            rate (Scalar, array-like, or float): Rate of nodal presession in radians/s.
            epoch Scalar, array-like, or float): Time in seconds TDB at which the `node`
                applies.
            reference (Frame or str): The Frame or the ID of the Frame describing the
                central planet of the inclined plane.
            despin (bool, optional): True for a nearly inertial frame, in which the x and
                yaxes vary as little as possible while the zaxis rotates; False for a
                frame in which the x axis is tied to the ascending node.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_INCLINED" to the Path ID of `reference`
                (if it has an ID).

        Raises:
            ValueError: If `inc`, `node`, `rate`, `epoch`, and `reference` cannot be
                broadcasted to the same shape.
        """

        self._inc = Scalar.as_scalar(inc).wod.as_readonly()
        self._node = Scalar.as_scalar(node).wod.as_readonly()
        self._rate = Scalar.as_scalar(rate).wod.as_readonly()
        self._epoch = Scalar.as_scalar(epoch).wod.as_readonly()
        self._despin = bool(despin)

        self._reference = Frame.as_wayframe(reference)
        self._origin = self._reference._origin
        self._shape = Qube.broadcasted_shape(self._inc, self._node, self._rate,
                                             self._epoch, self._reference._shape)

        if frame_id == '+' and self._reference._frame_id:
            frame_id = self._reference._frame_id + '_INCLINED'

        self._register(frame_id)
        mutable.refresh(self)

    def _refresh(self):
        self._spin1 = SpinFrame(self._node, self._rate, self._epoch, axis=2,
                                reference=self._reference)
        self._rotate = Rotation(self._inc, axis=0, reference=self._spin1)

        if self._despin:
            self._spin2 = SpinFrame(-self._node, -self._rate, self._epoch, axis=2,
                                    reference=self._rotate)
        else:
            self._spin2 = None

    def _wayframe_key(self):
        return (self._inc, self._node, self._rate, self._epoch, self._reference,
                self._despin)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._inc, self._node, self._rate, self._epoch, self._reference,
                self._despin, self.stripped_id)

    def __setstate__(self, state):
        (inc, node, rate, epoch, reference, despin, frame_id) = state
        self.__init__(inc, node, rate, epoch, reference=reference, despin=despin,
                      frame_id=frame_id)
        mutable.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=False):
        """Transform that rotates coordinates from the reference frame to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class InclinedFrame.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        xform = self._spin1.transform_at_time(time)
        xform = self._rotate.transform_at_time(time).rotate_transform(xform)

        if self._spin2:
            xform = self._spin2.transform_at_time(time).rotate_transform(xform)

        return xform

    def node_at_time(self, time, *, quick=False):
        """The vector defining the ascending node of this frame's XY plane relative to
        the XY frame of its reference.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class InclinedFrame.

        Returns:
            (Vector3): The unit vector pointing in the direction of the ascending node.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        # Locate the ascending nodes in the reference frame
        time = Scalar.as_scalar(time)
        return (self._node + self._rate * (time - self._epoch)) % Scalar.TWOPI

##########################################################################################

Frame._FRAME_SUBCLASSES.append(InclinedFrame)

##########################################################################################
