##########################################################################################
# oops/frame/navigation.py
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

    _WAYFRAMES = {}

    def __init__(self, arg, /, reference, *, freeze=False, frame_id=None, _matrix=None):
        """Constructor for a Navigation.

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

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
            ValueError: If `arg` does not provide either two or three angles.
        """

        # Linking to a frozen object yields a frozen object
        if isinstance(arg, str):
            arg = Frame.as_frame(arg)
        if isinstance(arg, Navigation) and arg.is_frozen:
            arg = arg._angles
            freeze = True

        if isinstance(arg, Navigation):
            self._link = arg
            self._link.refresh()
            self._angles = self._link._angles
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
            frame_id = self._reference._frame_id + '_NAV'

        self._register(frame_id)
        self._refresh(matrix=self._matrix)
        if freeze:
            self.freeze()

    def _wayframe_key(self):
        return (self._angles, self._reference, self._link)

    @property
    def angles(self):
        """The two or three rotation angles in radians, as a tuple."""
        self.refresh()
        return self._angles

    @property
    def link(self):
        """The object to which this one is linked, or None if it is unlinked."""
        return self._link

    def _source(self):
        """The original source of the rotation angles if this object is linked to another;
        otherwise, self.
        """
        return self._link._source() if self._link else self

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        if self._link:
            arg_str = self._link.show(level-1, skip)
        else:
            arg_str = str(self._angles)

        return (f'{name}({arg_str},\n'
                f'{blanks}{self._reference.show(level-1, skip)})')

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

        if self._link:
            self._link.set_params(params)
            self._angles = self._link._angles
        else:
            self._angles = tuple(params)

    @property
    def params(self):
        """The fittable parameters of this Navigation, as a tuple of rotation angles."""
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
                                    origin=self._origin)

    def _freeze(self):
        if self._link:
            self._angles = self._link._angles
            self._link = None
        self._reregister()

    @staticmethod
    def _rotmat(angle, axis):
        """A matrix that performs a rotation about a single specified axis.

        Parameters:
            angle (float): The angle of rotation in radians.
            axis (int): The axis of rotation: 0 for x, 1 for y, 2 for z.

        Returns:
            Matrix3: The 3x3 matrix describing this rotation.
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
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class Navigation.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Notes:
            A Navigation is a fixed Frame, so the Transform relative to the `reference`
            Frame is independent of time. The returned Transform always has the shape of
            this Frame, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(Navigation)

##########################################################################################
