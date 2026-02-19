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

    _WAYFRAMES = {}
    _XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, arg, /, axis, reference, *, freeze=False, frame_id=None):
        """Constructor for a Rotation Frame.

        Parameters:
            arg (Scalar, array-like, float or Rotation): The angle of rotation in radians,
                which can be multidimensional. Alternatively, if another Rotation is
                given, this object's rotation angle will always match that of the
                argument.
            axis (int): The rotation axis: 0, "x", or "X" for x; 1, "y", or "Y" for y; 2,
                "z", or "Z" for z.
            reference (frame or str): The Frame or the ID of the Frame relative to which
                this rotation is defined.
            freeze (bool, optional): True to return a frozen object; False to leave it
                fittable.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_ROTATED" to the ID of `reference` (if
                it has an ID).
        """

        # Linking to a frozen object yields a frozen object
        if isinstance(arg, str):
            arg = Frame.as_frame(arg)
        if isinstance(arg, Rotation) and arg.is_frozen:
            arg = arg._angle
            freeze = True

        if isinstance(arg, Rotation):
            self._link = arg
            self._angle_shape = self._link._angle_shape
            self._angle_mask = self._link._angle_mask
        else:
            self._angle = Scalar.as_scalar(arg).wod.as_readonly()
            self._angle_shape = self._angle.shape
            self._angle_mask = self._angle.mask
            self._link = None

        self._axis2 = Rotation._XYZDICT[axis]
        self._axis0 = (self._axis2 + 1) % 3
        self._axis1 = (self._axis2 + 2) % 3

        self._reference = Frame.as_wayframe(reference)
        self._origin = self._reference._origin
        self._shape = Qube.broadcasted_shape(self._angle, self._reference)

        if frame_id == '+' and self._reference._frame_id:
            frame_id = self._reference._frame_id + '_ROTATED'

        self._register(frame_id)
        self.refresh()
        if freeze:
            self.freeze()

    @property
    def angle(self):
        self.refresh()
        return self._angle

    def _source(self):
        """The original source of the time shift if this object is linked to another;
        otherwise, self.
        """
        return self._link._source() if self._link else self

    def _wayframe_key(self):
        if self.is_frozen:
            return (self._angle, self._reference)
        # Use id(self) to ensure that an unlinked frame has a unique key
        return (self._link or id(self), self._reference)

    ######################################################################################
    # Fittable interface
    ######################################################################################

    nparams = 1

    def _set_params(self, params):
        """Redefine the rotation angle of this Rotation object."""

        if self._link:
            self._link.set_params(params)
            self._angle = self._link._angle
        elif self._angle_shape == ():
            self._angle = Scalar(params[0], self._angle_mask)
        else:
            params = np.array(params).reshape(self._angle_shape)
            self._angle = Scalar(params, self._angle_mask)

    @property
    def params(self):
        if self._angle_shape == ():
            return (self._angle,)
        else:
            return tuple(self._angle.vals.ravel())

    def _refresh(self):
        if self._link:
            self._angle = self._link._angle
            self._matrix = self._link._matrix
        else:
            mat = np.zeros(self._shape + (3, 3))
            mat[..., self._axis2, self._axis2] = 1.
            mat[..., self._axis0, self._axis0] = np.cos(self._angle.vals)
            mat[..., self._axis0, self._axis1] = np.sin(self._angle.vals)
            mat[..., self._axis1, self._axis1] =  mat[..., self._axis0, self._axis0]
            mat[..., self._axis1, self._axis0] = -mat[..., self._axis0, self._axis1]
            self._matrix = Matrix3(mat, self._angle_mask)

        self._transform = Transform(self._matrix, Vector3.ZERO, self, self._reference,
                                    self._origin)

    def _freeze(self):
        self._reregister()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._angle, self._axis2, self._reference, self.stripped_id)

    def __setstate__(self, state):
        (angle, axis, reference, frame_id) = state
        self.__init__(angle, axis, reference, frame_id=frame_id)
        self.freeze()

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
                class Rotation.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            Rotation is a fixed frame, so the transform relative to the `reference` frame
            is independent of time. The returned Transform always has the shape of this
            object, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(Rotation)

##########################################################################################
