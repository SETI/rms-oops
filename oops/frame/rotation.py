##########################################################################################
# oops/frame/rotation.py: Subclass Rotation of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.fittable  import Fittable
from oops.frame     import Frame
from oops.transform import Transform


class Rotation(Frame, Fittable):
    """A Frame describing a fixed rotation about one axis of another frame."""

    _XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, arg, /, axis, reference, *, frame_id=None):
        """Constructor for a Rotation Frame.

        Parameters:
            arg (Scalar, array-like, float or Rotation): The angle of rotation in radians,
                which can be multidimensional. Alternatively, if another Rotation is
                given, this object's rotation angle will always match that of the other.
            axis (int): The rotation axis: 0 or "X" for x; 1 or "Y" for y, or 2 or "Z" for
                z.
            reference (frame or str): The frame or frame ID relative to which this
                rotation is defined.
            frame_id (str, optional): The frame ID to use; None to use a temporary ID.
        """

        if isinstance(arg, Rotation):
            self.link = arg
            self._angle_shape = self.link.angle_shape
            self._fittable_nparams = self.link._fittable_nparams
        else:
            self.angle = Scalar.as_scalar(arg)
            self._angle_shape = self.angle.shape
            self._fittable_nparams = self.angle.size
            self.link = None

        self.axis2 = Rotation._XYZDICT[axis]
        self.axis0 = (self.axis2 + 1) % 3
        self.axis1 = (self.axis2 + 2) % 3

        # Required attributes
        self.reference = Frame.as_wayframe(reference)
        self.origin = self.reference.origin
        self.shape = Qube.broadcasted_shape(self.angle, self.reference)
        self.frame_id = frame_id

        self.register()
        self._refresh()

    ######################################################################################
    # Fittable interface
    ######################################################################################

    def _set_params(self, params):
        """Redefine the rotation angle of this Rotation object."""

        if self.link:
            self.link.set_params(params)
            self.angle = self.link.angle
        elif self._angle_shape == ():
            self.angle = Scalar(params[0], self.angle.mask)
        else:
            params = np.array(params).reshape(self._angle_shape)
            self.angle = Scalar(params, self.angle.mask)

    @property
    def _params(self):
        return (self.angle,)

    def _refresh(self):
        """Update the internals."""

        if self.link:
            self.angle = self.link.angle
            self._matrix = self.link._matrix
        else:
            mat = np.zeros(self.shape + (3, 3))
            mat[..., self.axis2, self.axis2] = 1.
            mat[..., self.axis0, self.axis0] = np.cos(self.angle.vals)
            mat[..., self.axis0, self.axis1] = np.sin(self.angle.vals)
            mat[..., self.axis1, self.axis1] =  mat[..., self.axis0, self.axis0]
            mat[..., self.axis1, self.axis0] = -mat[..., self.axis0, self.axis1]
            self._matrix = Matrix3(mat, self.angle.mask)

        self._transform = Transform(self._matrix, Vector3.ZERO, self, self.reference,
                                    self.origin)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self.angle, self.axis2, Frame.as_primary_frame(self.reference),
                self._state_id())

    def __setstate__(self, state):
        (angle, axis, reference, frame_id) = state
        self.__init__(angle, axis, reference, frame_id=frame_id)
        self.freeze()

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

        Notes:
            Rotation is a fixed frame, so the transform relative to the `reference` frame
            is independent of time.
        """

        return self._transform

##########################################################################################
