##########################################################################################
# oops/frame/twovectorframe.py: Subclass TwoVectorFrame of class Frame
##########################################################################################

from polymath       import Matrix3, Qube, Vector3
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform


class TwoVectorFrame(Frame):
    """A Frame subclass describing a frame that is fixed relative to another frame.

    It is described by two vectors. The first vector is one axis of the frame and the
    second vector points in the half-plane of another axis.
    """

    _FRAME_IDS = {}
    _XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, reference, vector1, axis1, vector2, axis2, frame_id='+'):
        """Constructor for a TwoVectorFrame.

        Parameters:
            reference (Frame): The frame relative to which this frame is defined.
            vector1 (Vector3 or array-like): Vector describing an axis.
            axis1 (str): The axis defined by the first vector: 0 or "X" for x; 1 or "Y"
                for y, or 2 or "Z" for z.
            vector1 (Vector3 or array-like): A Vector which, along with vector1, defines
                the half-plane in which a second axis falls.
            axis1 (str): "X", "Y", or "Z", indicating the axis defined by the second
                vector.
            frame_id (str, optional): The ID under which to register the frame; None to
                leave it unregistered. As a special case, at value of "+" alone is
                replaced by the ID of `reference` plus "_TWOVECTOR". If text follows the
                "+", the new ID is the ID of `reference` followed by "_" and this text.
        """

        self.vector1 = Vector3.as_vector3(vector1)
        self.vector2 = Vector3.as_vector3(vector2)
        self.axis1 = TwoVectorFrame._XYZDICT[axis1]
        self.axis2 = TwoVectorFrame._XYZDICT[axis2]

        # Required attributes
        self.reference = Frame.as_wayframe(reference)
        self.origin = self.reference.origin
        self.shape = Qube.broadcasted_shape(self.vector1, self.vector2, self.reference)

        if frame_id is None:
            self.frame_id = Frame._recover_frame_id(frame_id)
        elif frame_id.startswith('+') and len(frame_id) > 1:
            self.frame_id = self.reference.frame_id + '_' + frame_id[1:]
        elif frame_id == '+':
            self.frame_id = self.reference.frame_id + '_TWOVECTOR'
        else:
            self.frame_id = frame_id

        # Register if necessary
        self.register()
        self._refresh()
        self._cache_id()

    def _refresh(self):
        matrix = Matrix3.twovec(self.vector1, self.axis1, self.vector2, self.axis2)
        self._transform = Transform(matrix, Vector3.ZERO, self, self.reference)
        z_axis = matrix.row_vector(2, classes=[Vector3])
        self._node = Vector3.ZAXIS.ucross(z_axis)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.reference, self.vector1, self.axis1, self.vector2, self.axis2)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (Frame.as_primary_frame(self.reference),  self.vector1, self.axis1,
                self.vector2, self.axis2, self._state_id())

    def __setstate__(self, state):
        (frame, vector1, axis1, vector2, axis2, frame_id) = state
        self.__init__(frame, vector1, axis1, vector2, axis2, frame_id=frame_id)
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
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            TwoVector is a fixed frame, so the transform relative to the `reference` frame
            is independent of time.
        """

        return self._transform

    def node_at_time(self, time, quick={}):
        """The vector defining the ascending node of this frame's XY plane relative to
        the XY frame of its reference.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Vector3): The unit vector pointing in the direction of the ascending node.

        Notes:
            TwoVector is a fixed frame, so its node vector relative to the `reference`
            frame is independent of time.
        """

        return self._node

##########################################################################################
