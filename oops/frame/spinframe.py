##########################################################################################
# oops/frame/spinframe.py: Subclass SpinFrame of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform


class SpinFrame(Frame):
    """A Frame subclass describing a frame in uniform rotation about one axis of another
    frame.

    It can be created without a frame_id, reference_id or origin_id; in this case it is
    not registered and can therefore be used as a component of another frame.
    """

    _FRAME_IDS = {}

    def __init__(self, offset, rate, epoch, axis, reference, frame_id=None):
        """Constructor for a Spin Frame.

        Input:
            offset (Scalar, array-like, or float): The angular offset of the frame at the
                epoch, in radians.
            rate (Scalar, array-like, or float): The rotation rate of the frame in
                radians/second.
            epoch (Scalar, array-like, or float): The time TDB at which the frame is
                defined.
            axis (int): The rotation axis: 0 for x, 1 for y, 2 for z.
            reference (Frame or str): The frame or ID relative to which this frame is
                defined.
            frame_id (str, optional): The ID to use; None to leave the frame unregistered.

        Notes:
            The rate, offset and epoch can be Scalar values, in which case the shape of
            the SpinFrame is defined by broadcasting the shapes of these Scalars.
        """

        self.offset = Scalar.as_scalar(offset)
        self.rate = Scalar.as_scalar(rate)
        self.epoch = Scalar.as_scalar(epoch)

        self.axis2 = axis
        self.axis0 = (self.axis2 + 1) % 3
        self.axis1 = (self.axis2 + 2) % 3

        self.shape = Qube.broadcasted_shape(self.rate, self.offset, self.epoch)
        omega_vals = np.zeros(list(self.shape) + [3])
        omega_vals[..., self.axis2] = self.rate.vals
        self.omega = Vector3(omega_vals, self.rate.mask)

        # Required attributes
        self.reference = Frame.as_wayframe(reference)
        self.origin = self.reference.origin or Frame.PATH_CLASS.SSB
        self.frame_id = self._recover_id(frame_id)

        # Update wayframe and frame_id; register if not temporary
        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.offset, self.rate, self.epoch, self.axis2, self.reference)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.offset, self.rate, self.epoch, self.axis2,
                Frame.as_primary_frame(self.reference), self._state_id())

    def __setstate__(self, state):
        (offset, rate, epoch, axis, reference, frame_id) = state
        self.__init__(offset, rate, epoch, axis, reference, frame_id=frame_id)
        Fittable_.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, quick={}):
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
        """

        time = Scalar.as_scalar(time)
        angle = (time - self.epoch) * self.rate + self.offset

        mat = np.zeros(list(angle.shape) + [3, 3])
        mat[..., self.axis2, self.axis2] = 1.
        mat[..., self.axis0, self.axis0] = np.cos(angle.values)
        mat[..., self.axis1, self.axis1] = mat[..., self.axis0, self.axis0]
        mat[..., self.axis0, self.axis1] = np.sin(angle.values)
        mat[..., self.axis1, self.axis0] = -mat[...,self.axis0, self.axis1]

        matrix = Matrix3(mat, angle.mask)
        return Transform(matrix, self.omega, self.wayframe, self.reference, self.origin)

##########################################################################################
