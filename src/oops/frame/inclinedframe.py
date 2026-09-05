##########################################################################################
# oops/frame/inclinedframe.py
##########################################################################################

from polymath             import Qube, Scalar
from oops.frame           import Frame
from oops.frame.rotation  import Rotation
from oops.frame.spinframe import SpinFrame


class InclinedFrame(Frame):
    """A Frame inclined to the equator of another Frame.

    It is defined by an inclination, a node at epoch, and a nodal regression rate. This
    Frame is oriented to be "nearly inertial," meaning that a longitude in the new Frame
    is determined by measuring from the reference longitude in the reference Frame, along
    that Frame's equator to the ascending node, and thence along the ascending node.
    """

    _WAYFRAMES = {}

    _USE_QUICKFRAMES = True     # nodal precession is always slow

    def __init__(self, inc, node, rate, epoch, *, despin=True, reference=None,
                 frame_id=None):
        """Constructor for an InclinedFrame.

        Parameters:
            inc (Scalar): The inclination angle in radians.
            node (Scalar): The longitude of the ascending node at `epoch`, in radians.
            rate (Scalar): The rate of nodal precession in radians/second.
            epoch (Scalar): The time in seconds TDB at which `node` applies.
            despin (bool, optional): True for a nearly inertial Frame, in which the *x*-
                and *y*-axes vary as little as possible while the *z*-axis rotates; False
                for a Frame in which the *x*-axis is tied to the ascending node.
            reference (Frame or str, optional): The Frame or the ID of the Frame
                describing the central planet of the inclined plane; None for J2000.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_INCLINED" to the ID of `reference` (if
                it has an ID).

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
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
        self.refresh()

    def _refresh(self):
        self._spin1 = SpinFrame(self._node, self._rate, self._epoch, axis=2,
                                reference=self._reference)
        self._rotate = Rotation(self._inc, axis=0, reference=self._spin1)
        self._rotate.freeze()

        if self._despin:
            self._spin2 = SpinFrame(-self._node, -self._rate, self._epoch, axis=2,
                                    reference=self._rotate)
        else:
            self._spin2 = None

    def _wayframe_key(self):
        return (self._inc, self._node, self._rate, self._epoch, self._reference,
                self._despin)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        if self._reference == Frame.J2000:
            ref_str = '"J2000"'
        else:
            ref_str = self._reference.show(level-1, skip)

        return (f'{name}(inc = {self._inc.mvals},\n'
                f'{blanks}node = {self._node.mvals},\n'
                f'{blanks}rate = {self._rate.mvals},\n'
                f'{blanks}epoch = {self._epoch.mvals},\n'
                f'{blanks}reference = {ref_str},\n'
                f'{blanks}despin = {self._despin})')

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._inc, self._node, self._rate, self._epoch, self._reference,
                self._despin, self.stripped_id)

    def __setstate__(self, state):
        (inc, node, rate, epoch, reference, despin, frame_id) = state
        self.__init__(inc, node, rate, epoch, reference=reference, despin=despin,
                      frame_id=frame_id)
        self.freeze()

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=False):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): Ignored by class InclinedFrame.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        xform = self._spin1.transform_at_time(time)
        xform = self._rotate.transform_at_time(time).rotate_transform(xform)

        if self._spin2:
            xform = self._spin2.transform_at_time(time).rotate_transform(xform)

        return xform

    def node_at_time(self, time, *, quick=False):
        """The angle from the reference Frame's *x*-axis to this Frame's ascending node.

        The angle is measured within the *xy* plane of the reference frame, to the
        ascending node of this Frame's *xy* plane.

        Values always fall between 0 and 2*pi.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): Ignored by class InclinedFrame.

        Returns:
            Scalar: At the specified times, the angle from the reference Frame's *x*-axis,
            along its *xy* plane, to the ascending node of this Frame's *xy* plane.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        # Locate the ascending nodes in the reference frame
        time = Scalar.as_scalar(time)
        return (self._node + self._rate * (time - self._epoch)) % Scalar.TWOPI

##########################################################################################

Frame._FRAME_SUBCLASSES.append(InclinedFrame)

##########################################################################################
