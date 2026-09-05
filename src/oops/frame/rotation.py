##########################################################################################
# oops/frame/rotation.py
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.fittable  import Fittable
from oops.frame     import Frame
from oops.transform import Transform


class Rotation(Frame, Fittable):
    """A Frame subclass describing a fixed rotation about one axis of another Frame."""

    _WAYFRAMES = {}
    _XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, arg, /, axis, reference, *, freeze=False, frame_id=None):
        """Constructor for a Rotation.

        Parameters:
            arg (Scalar, array-like, float, or Rotation): The angle of rotation in
                radians, which can be multidimensional. Alternatively, if another Rotation
                is given, this object's rotation angle will always match that of the
                argument.
            axis (int or str): The rotation axis: 0, "x", or "X" for *x*; 1, "y", or "Y"
                for *y*; 2, "z", or "Z" for *z*.
            reference (Frame or str): The Frame or the ID of the Frame relative to which
                this rotation is defined.
            freeze (bool, optional): True to return a frozen object; False to leave it
                fittable.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_ROTATED" to the ID of `reference` (if
                it has an ID).

        Raises:
            KeyError: If `axis` is not a recognized axis, or if `reference` is an ID
                string that has not been registered.
        """

        # Linking to a frozen object yields a frozen object
        if isinstance(arg, str):
            arg = Frame.as_frame(arg)
        if isinstance(arg, Rotation) and arg.is_frozen:
            arg = arg._angle
            freeze = True

        if isinstance(arg, Rotation):
            self._link = arg
            self._link.refresh()
            self._angle = self._link._angle
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

    def _wayframe_key(self):
        return (self._angle, self._axis2, self._reference, self._link)

    @property
    def angle(self):
        """The angle of rotation in radians, as a Scalar."""
        self.refresh()
        return self._angle

    def _source(self):
        """The original source of the rotation angle, or self if there is none.
        """
        return self._link._source() if self._link else self

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        if hasattr(self._link, 'show'):
            parts = [self._link.show(level-1, skip),
                     str(self._axis2),
                     self._reference.show(level-1, skip)]
        else:
            parts = [f'{self._angle.mvals}, {self._axis2}',
                     self._reference.show(level-1, skip)]

        return f'{name}(' + f'\n{blanks}'.join(parts) + ')'

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
        """The fittable parameters of this Rotation, as a tuple of rotation angles."""
        if self._angle_shape == ():
            return (self._angle.vals,)
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
                                    origin=self._origin)

    def _freeze(self):
        if self._link:
            self._angle = self._link._angle
            self._link = None
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
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): Ignored by class Rotation.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Notes:
            A Rotation is a fixed Frame, so the Transform relative to the `reference`
            Frame is independent of time. The returned Transform always has the shape of
            this Frame, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(Rotation)

##########################################################################################
