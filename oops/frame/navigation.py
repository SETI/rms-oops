##########################################################################################
# oops/frame/navigation.py: Fittable subclass Navigation of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Vector3
from oops.fittable  import Fittable
from oops.frame     import Frame
from oops.transform import Transform


class Navigation(Frame, Fittable):
    """A Frame subclass describing a fittable, fixed offset from another frame, defined by
    two or three rotation angles.
    """

    _FRAME_IDS = {}

    def __init__(self, arg, /, reference, *, freeze=False, frame_id=None, _matrix=None):
        """Constructor for a Navigation Frame.

        Parameters:
            arg (array-like or Navigation): Two or three angles of rotation in radians.
                The order of the rotations is about the y, x, and (optionally) z axes.
                These angles rotate a vector in the reference frame into this frame.
                Alternatively, specify another Navigation object and this object will be
                linked to that one, meaning that the rotation angles will always match.
            reference (Frame or str): The Frame or the ID of the Frame relative to which
                this navigation applies.
            freeze (bool, optional): True to return a frozen object; False to leave it
                fittable.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_NAV" to the ID of `reference` (if it
                has an ID).
            _matrix (Matrix3, optional): A 3x3 matrix, used internally, to speed up the
                copying of Navigation objects. If provided, it must contain the Matrix3
                object that performs the defined rotation.
        """

        # Linking to a frozen object yields a frozen object
        if isinstance(arg, str):
            arg = Frame.as_frame(frame)
        if isinstance(arg, Navigation) and arg.is_frozen:
            arg = arg._angles
            freeze = True

        if isinstance(arg, Navigation):
            self._link = arg
            self._link.refresh()
            self._matrix = None
        else:
            self._angles = tuple(arg)
            self._link = None
            self._matrix = _matrix

        self.nparams = len(self._angles)
        if self.nparams not in {2, 3}:
            raise ValueError('two or three Navigation angles must be provided')

        self._reference = Frame.as_wayframe(reference)
        self._origin = self._reference._origin
        self._shape = self._reference._shape

        if frame_id == '+' and self._reference._frame_id:
            if self._epoch is None:
                frame_id = self._reference._frame_id + '_NAV'

        self._register(frame_id)
        self._refresh(matrix=self._matrix)
        if freeze:
            self.freeze()

    @property
    def angles(self):
        self.refresh()
        return self._angles

    def _source(self):
        """The original source of the time shift if this object is linked to another;
        otherwise, self.
        """
        return self._link._source() if self._link else self

    def _wayframe_key(self):
        if self.is_frozen:
            return (self._angles, self._reference)
        # Use id(self) to ensure that an unlinked frame has a unique key
        return (self._link or id(self), self._reference)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._angles, self._reference, self.stripped_id)

    def __setstate__(self, state):
        (angles, reference, frame_id) = state
        self.__init__(angles, reference, frame_id=frame_id)
        self.freeze()

    ######################################################################################
    # Fittable interface
    ######################################################################################

    def _set_params(self, params):
        """Redefine the navigation angles."""

        if self.link:
            self._link.set_params(params)
            self._angles = self._link._angles
        else:
            self._angles = params

    @property
    def params(self):
        return self._angles

    def _refresh(self, matrix=None):
        if self._link:
            self._angles = self._link._angles
            self._matrix = self._link._matrix
        elif matrix is None:
            matrix = Navigation._rotmat(self._angles[0], 1)
            matrix = Navigation._rotmat(self._angles[1], 0) * matrix
            if self.nparams > 2 and self._angles[2] != 0.:
                matrix = Navigation._rotmat(self._angles[2], 2) * matrix
            self._matrix = matrix

        self._transform = Transform(self._matrix, Vector3.ZERO, self, self._reference,
                                    self._origin)

    def _freeze(self):
        if self._link:
            self._angles = self._link._angles
            self._link = None
        self._reregister()

    @staticmethod
    def _rotmat(angle, axis):
        """Internal function to return a matrix that performs a rotation about a single
        specified axis.
        """

        axis2 = axis
        axis0 = (axis2 + 1) % 3
        axis1 = (axis2 + 2) % 3

        mat = np.zeros((3,3))
        mat[axis2, axis2] = 1.
        mat[axis0, axis0] = np.cos(angle)
        mat[axis0, axis1] = np.sin(angle)
        mat[axis1, axis1] =  mat[axis0, axis0]
        mat[axis1, axis0] = -mat[axis0, axis1]

        return Matrix3(mat)

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
                class Navigation.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            Navigation is a fixed frame, so the transform relative to the `reference`
            frame is independent of time. The returned Transform always has the shape of
            this object, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(Navigation)

##########################################################################################
