##########################################################################################
# oops/frame/spinframe.py
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform


class SpinFrame(Frame):
    """A Frame subclass describing a Frame in uniform rotation about one axis of another
    Frame.

    It can be created without a `frame_id`, in which case it is left unregistered and can
    therefore be used as a component of another Frame.
    """

    _WAYFRAMES = {}
    # _USE_QUICKFRAMES is False because rotation might be rapid and the calculation is
    # already fairly quick.

    def __init__(self, offset, rate, epoch, axis, reference, *, frame_id=None):
        """Constructor for a SpinFrame.

        Parameters:
            offset (Scalar, array-like, or float): The rotation angle in radians at
                `epoch`.
            rate (Scalar, array-like, or float): The rotation rate in radians/second.
            epoch (Scalar, array-like, or float): The time in seconds TDB at which
                `offset` applies.
            axis (int): The rotation axis: 0 for x, 1 for y, 2 for z.
            reference (Frame or str): The Frame or the ID of the Frame relative to which
                this rotation is defined.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_SPIN" to the ID of `reference` (if it
                has an ID).

        Notes:
            `offset`, `rate`, and `epoch` can be Scalars, in which case the shape of the
            SpinFrame is defined by broadcasting their shapes together with that of
            `reference`.

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
            ValueError: If `offset`, `rate`, `epoch`, and `reference` cannot be
                broadcasted to the same shape.
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
        self.refresh()

    def _wayframe_key(self):
        return (self._offset, self._rate, self._epoch, self._axis2, self._reference)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        return (f'{name}(offset = {self._offset.mvals},\n'
                f'{blanks}rate = {self._rate.mvals},\n'
                f'{blanks}epoch = {self._epoch.mvals},\n'
                f'{blanks}axis = {self._axis2},\n'
                f'{blanks}reference = {self._reference.show(level-1, skip)})')

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._offset, self._rate, self._epoch, self._axis2, self._reference,
                self.stripped_id)

    def __setstate__(self, state):
        (offset, rate, epoch, axis, reference, frame_id) = state
        self.__init__(offset, rate, epoch, axis, reference, frame_id=frame_id)
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
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored for
                class SpinFrame.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

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
