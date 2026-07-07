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

    def __init__(self, arg, /, reference, *, frame_id=None, override=False, _matrix=None):
        """Constructor for a Navigation Frame.

        Parameters:
            arg (array-like or Navigation): Two or three angles of rotation in radians.
                The order of the rotations is about the y, x, and (optionally) z axes.
                These angles rotate a vector in the reference frame into this frame.
                Alternatively, specify another Navigation object and this object will be
                linked to that one, meaning that the rotation angles will always match.
            reference (Frame or str): The frame or frame ID relative to which this
                navigation applies.
            frame_id (str, optional): The frame ID to use; None to use a temporary ID.
            override (bool, optional): True to override a pre-existing frame with the same
                ID.
            _matrix (Matrix3, optional): A 3x3 matrix, used internally, to speed up the
                copying of Navigation objects. If not None, it must contain the Matrix3
                object that performs the defined rotation.
        """

        if isinstance(arg, Navigation):
            self.link = arg
            self.link.refresh()
            self.angles = self.link.angles
            self._matrix = self.link._matrix
        else:
            self.angles = tuple(arg)
            self.link = None
            self._matrix = _matrix

        self._fittable_nparams = len(self.angles)
        if self._fittable_nparams not in {2, 3}:
            raise ValueError('two or three Navigation angles must be provided')

        self.reference = Frame.as_wayframe(reference)
        self.origin = self.reference.origin
        self.shape = self.reference.shape
        self.frame_id = Frame._recover_id(frame_id)

        # Update wayframe and frame_id; register if not temporary
        self.register(override=override)

        # Fill in transform (_after_ registration)
        self._refresh(matrix=self._matrix)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self.angles, Frame.as_primary_frame(self.reference), self._state_id())

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
            self.link.set_params(params)
            self.angles = self.link.angles
        else:
            self.angles = params

    @property
    def _params(self):
        return self.angles

    def _refresh(self, matrix=None):
        """Update the internals."""

        if self.link:
            self.angles = self.link.angles
            self._matrix = self.link._matrix
        elif matrix is None:
            matrix = Navigation._rotmat(self.angles[0], 1)
            matrix = Navigation._rotmat(self.angles[1], 0) * matrix
            if self._fittable_nparams > 2 and self.angles[2] != 0.:
                matrix = Navigation._rotmat(self.angles[2], 2) * matrix
            self._matrix = matrix

        self._transform = Transform(matrix, Vector3.ZERO, self, self.reference,
                                    self.origin)

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
            Navigation is a fixed frame, so the transform relative to the `reference`
            frame is independent of time.
        """

        return self._transform

##########################################################################################
