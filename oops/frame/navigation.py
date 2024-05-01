################################################################################
# oops/frame/navigation.py: Fittable subclass Navigation of class Frame
################################################################################

import numpy as np

from polymath       import Matrix3, Vector, Vector3
from oops.fittable  import Fittable
from oops.frame     import Frame
from oops.transform import Transform

class Navigation(Frame, Fittable):
    """A Frame subclass describing a fittable, fixed offset from another frame,
    defined by two or three rotation angles.
    """

    # Note: Navigation frames are not generally re-used, so their IDs are
    # expendable. Frame IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, angles, reference, frame_id=None, override=False,
                       _matrix=None):
        """Constructor for a Navigation Frame.

        Input:
            angles      two or three angles of rotation in radians. The order of
                        the rotations is about the y, x, and (optionally) z
                        axes. These angles rotate a vector in the reference
                        frame into this frame.
            reference   the frame or frame ID relative to which this rotation is
                        defined.
            frame_id    the ID to use; None to use a temporary ID.
            override    True to override a pre-existing frame with the same ID.
            _matrix     an optional 3x3 matrix, used internally, to speed up the
                        copying of Navigation objects. If not None, it must
                        contain the Matrix3 object that performs the defined
                        rotation.
        """

        if isinstance(angles, Vector):
            angles = angles.vals

        self.angles = np.array(angles)
        if self.angles.shape not in ((2,),(3,)):
            raise ValueError('two or three Navigation angles must be provided')

        self.cache = {}
        self.param_name = 'angles'
        self.nparams = self.angles.shape[0]

        if _matrix is None:
            _matrix = Navigation._rotmat(self.angles[0],1)
            _matrix = Navigation._rotmat(self.angles[1],0) * _matrix

            if self.nparams > 2 and self.angles[2] != 0.:
                _matrix = Navigation._rotmat(self.angles[2], 2) * _matrix

        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.frame_id  = frame_id
        self.shape     = self.reference.shape
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register(override=override)

        # Fill in transform (_after_ registration)
        self.transform = Transform(_matrix, Vector3.ZERO,
                                   self, self.reference, self.origin)

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.angles, Frame.as_primary_frame(self.reference))

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def _rotmat(angle, axis):
        """Internal function to return a matrix that performs a rotation about
        a single specified axis.
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

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """The Transform to the given Frame at a specified Scalar of
        times.
        """

        return self.transform

    ############################################################################
    # Fittable interface
    ############################################################################

    def set_params_new(self, params):
        """Redefines the Fittable object, using this set of parameters. Unlike
        method set_params(), this method does not check the cache first.
        Override this method if the subclass should use a cache.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        params = np.array(params).copy()
        if self.angles.shape != params.shape:
            raise ValueError('new parameter shape does not match original')

        self.angles = params

        matrix = Navigation._rotmat(self.angles[0],1)
        matrix = Navigation._rotmat(self.angles[1],0) * matrix

        if self.nparams > 2 and self.angles[2] != 0.:
            matrix = Navigation._rotmat(self.angles[2],2) * matrix

        self.transform = Transform(matrix, Vector3.ZERO, self,
                                   self.reference, self.origin)

    def copy(self):
        """A deep copy of the given object. The copy can be safely modified
        without affecting the original.
        """

        return Navigation(self.angles.copy(), self.reference,
                          matrix=self.transform.matrix.copy())

################################################################################
