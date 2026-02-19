##########################################################################################
# oops/frame/spinframe.py: Subclass SpinFrame of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class SpinFrame(Frame):
    """A Frame subclass describing a frame in uniform rotation about one axis of another
    frame.

    It can be created without a frame_id, reference_id or origin_id; in this case it is
    not registered and can therefore be used as a component of another frame.
    """

    _WAYFRAMES = {}

    def __init__(self, offset, rate, epoch, axis, reference, *, frame_id=None):
        """Constructor for a Spin Frame.

        Input:
            offset (Scalar, array-like, or float): The angular offset of the frame at the
                epoch, in radians.
            rate (Scalar, array-like, or float): The rotation rate of the frame in
                radians/second.
            epoch (Scalar, array-like, or float): The time TDB at which the frame is
                defined.
            axis (int): The rotation axis: 0 for x, 1 for y, 2 for z.
            reference (Frame or str): The Frame or the ID of the Frame relative to which
                this Frame is defined.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_SPIN" to the ID of `reference` (if it
                has an ID).

        Raises:
            ValueError: If `offset`, `rate`, `epoch`, and  and epoch can be Scalar values,
                in which case the shape of the SpinFrame is defined by broadcasting the
                shapes of these Scalars.
        """

        self._offset = Scalar.as_scalar(offset).wod.as_readonly()
        self._rate = Scalar.as_scalar(rate).wod.as_readonly()
        self._epoch = Scalar.as_scalar(epoch).wod.as_readonly()

        self._axis2 = axis
        self._axis0 = (self._axis2 + 1) % 3
        self._axis1 = (self._axis2 + 2) % 3

        self._reference = Frame.as_wayframe(reference)
        self._shape = Qube.broadcasted_shape(self._rate, self._offset, self._epoch,
                                             self._reference._shape)
        omega_vals = np.zeros(list(self._shape) + [3])
        omega_vals[..., self._axis2] = self._rate.vals
        self._omega = Vector3(omega_vals, self._rate.mask)
        self._origin = self._reference._origin or Frame._Path.SSB

        if frame_id == '+' and self._reference._frame_id:
            frame_id = self._reference._frame_id + '_SPIN'

        self._register(frame_id)
        mutable.refresh(self)

    def _wayframe_key(self):
        return (self._offset, self._rate, self._epoch, self._axis2, self._reference)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._offset, self._rate, self._epoch, self._axis2, self._reference,
                self.stripped_id)

    def __setstate__(self, state):
        (offset, rate, epoch, axis, reference, frame_id) = state
        self.__init__(offset, rate, epoch, axis, reference, frame_id=frame_id)
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
                Use False to disable the use of QuickPaths and QuickFrames. Ignored for
                class SpinFrame.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        time = Scalar.as_scalar(time)
        angle = (time - self._epoch) * self._rate + self._offset

        mat = np.zeros(angle._shape + (3, 3))
        mat[..., self._axis2, self._axis2] = 1.
        mat[..., self._axis0, self._axis0] = np.cos(angle.vals)
        mat[..., self._axis1, self._axis1] = mat[..., self._axis0, self._axis0]
        mat[..., self._axis0, self._axis1] = np.sin(angle.vals)
        mat[..., self._axis1, self._axis0] = -mat[..., self._axis0, self._axis1]

        matrix = Matrix3(mat, angle.mask)
        return Transform(matrix, self._omega, self, self._reference, self._origin)

##########################################################################################

Frame._FRAME_SUBCLASSES.append(SpinFrame)

##########################################################################################
