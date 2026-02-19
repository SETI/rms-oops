##########################################################################################
# oops/frame/twovectorframe.py: Subclass TwoVectorFrame of class Frame
##########################################################################################

from polymath       import Matrix3, Qube, Vector3
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class TwoVectorFrame(Frame):
    """A Frame subclass describing a frame that is fixed relative to another frame.

    It is described by two vectors. The first vector is one axis of the frame and the
    second vector points in the half-plane of another axis.
    """

    _WAYFRAMES = {}
    _XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, reference, vector1, axis1, vector2, axis2, *, frame_id=None):
        """Constructor for a TwoVectorFrame.

        Parameters:
            reference (Frame): The Frame or the ID of the Frame relative to which this
                Frame is defined.
            vector1 (Vector3 or array-like): Vector describing an axis.
            axis1 (int or str): The axis defined by the first vector: 0, "x", or "X" for
                x; 1, "y", or "Y" for y; 2, "z", or "Z" for z.
            vector1 (Vector3 or array-like): A Vector which, along with `vector1`, defines
                the half-plane in which a second axis falls.
            axis1 (int or str): The axis defined by the second vector: 0, "x", or "X" for
                x; 1, "y", or "Y" for y; 2, "z", or "Z" for z.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered.

        Raises:
            ValueError: If `reference`, `vector1`, `axis1`, `vector2`, and `axis2` cannot
                be broadcasted to the same shape.
        """

        self._vector1 = Vector3.as_vector3(vector1).wod.as_readonly()
        self._vector2 = Vector3.as_vector3(vector2).wod.as_readonly()
        self._axis1 = TwoVectorFrame._XYZDICT[axis1]
        self._axis2 = TwoVectorFrame._XYZDICT[axis2]

        self._reference = Frame.as_wayframe(reference)
        self._origin = self._reference._origin
        self._shape = Qube.broadcasted_shape(self._vector1, self._vector2,
                                             self._reference)

        self._register(frame_id)
        mutable.refresh(self)

    def _refresh(self):
        matrix = Matrix3.twovec(self._vector1, self._axis1, self._vector2, self._axis2)
        self._transform = Transform(matrix, Vector3.ZERO, self, self._reference)
        z_axis = matrix.row_vector(2, classes=[Vector3])
        self._node = Vector3.ZAXIS.ucross(z_axis)

    def _wayframe_key(self):
        return (self._reference, self._vector1, self._axis1, self._vector2, self._axis2)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._reference,  self._vector1, self._axis1, self._vector2, self._axis2,
                self.stripped_id)

    def __setstate__(self, state):
        (frame, vector1, axis1, vector2, axis2, frame_id) = state
        self.__init__(frame, vector1, axis1, vector2, axis2, frame_id=frame_id)
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
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class TwoVectorFrame.

        Returns:
            (Transform): The Transform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            TwoVector is a fixed frame, so the Transform relative to the `reference` Frame
            is independent of time. The returned Transform always has the shape of this
            object, regardless of the shape of `time`.
        """

        return self._transform

    def node_at_time(self, time, *, quick=False):
        """The vector defining the ascending node of this frame's XY plane relative to
        the XY frame of its reference.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class TwoVectorFrame.

        Returns:
            (Vector3): The unit vector pointing in the direction of the ascending node.

        Notes:
            TwoVector is a fixed Frame, so its node vector relative to the `reference`
            Frame is independent of time. The returned Transform always has the shape of
            this object, regardless of the shape of `time`.
        """

        return self._node

##########################################################################################

Frame._FRAME_SUBCLASSES.append(TwoVectorFrame)

##########################################################################################
