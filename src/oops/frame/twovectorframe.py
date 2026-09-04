##########################################################################################
# oops/frame/twovectorframe.py
##########################################################################################

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform


class TwoVectorFrame(Frame):
    """A Frame subclass describing a Frame that is fixed relative to another Frame.

    It is described by two vectors. The first vector is one axis of the Frame and the
    second vector points in the half-plane of another axis.
    """

    _WAYFRAMES = {}
    _XYZDICT = {'X': 0, 'Y': 1, 'Z': 2, 'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, reference, vector1, axis1, vector2, axis2, *, frame_id=None):
        """Constructor for a TwoVectorFrame.

        Parameters:
            reference (Frame or str): The Frame or the ID of the Frame relative to which
                this Frame is defined.
            vector1 (Vector3 or array-like): A vector describing an axis.
            axis1 (int or str): The axis defined by the first vector: 0, "x", or "X" for
                x; 1, "y", or "Y" for y; 2, "z", or "Z" for z.
            vector2 (Vector3 or array-like): A vector which, along with `vector1`, defines
                the half-plane in which a second axis falls.
            axis2 (int or str): The axis defined by the second vector: 0, "x", or "X" for
                x; 1, "y", or "Y" for y; 2, "z", or "Z" for z.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered.

        Raises:
            KeyError: If `axis1` or `axis2` is not a recognized axis, or if `reference` is
                an ID string that has not been registered.
            ValueError: If `reference`, `vector1`, and `vector2` cannot be broadcasted to
                the same shape.
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
        self.refresh()

    def _refresh(self):
        matrix = Matrix3.twovec(self._vector1, self._axis1, self._vector2, self._axis2)
        self._transform = Transform(matrix, Vector3.ZERO, self, self._reference)
        z_axis = matrix.row_vector(2, classes=[Vector3])
        (x, y, _) = z_axis.to_scalars()
        self._node = (y.arctan2(x) + Scalar.HALFPI) % Scalar.TWOPI

    def _wayframe_key(self):
        return (self._reference, self._vector1, self._axis1, self._vector2, self._axis2)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        return (f'{name}({self._reference.show(level-1, skip)},\n'
                f'{blanks}{self._vector1.mvals}, {self._axis1},\n'
                f'{blanks}{self._vector2.mvals}, {self._axis2})')

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._reference,  self._vector1, self._axis1, self._vector2, self._axis2,
                self.stripped_id)

    def __setstate__(self, state):
        (frame, vector1, axis1, vector2, axis2, frame_id) = state
        self.__init__(frame, vector1, axis1, vector2, axis2, frame_id=frame_id)
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
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class TwoVectorFrame.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Notes:
            A TwoVectorFrame is a fixed Frame, so the Transform relative to the
            `reference` Frame is independent of time. The returned Transform always has
            the shape of this Frame, regardless of the shape of `time`.
        """

        return self._transform

    def node_at_time(self, time, *, quick=False):
        """The angle from the reference Frame's X-axis to this Frame's ascending node.

        The angle is measured within the X-Y plane of the reference frame, to the
        ascending node of this Frame's X-Y plane.

        Values always fall between 0 and 2*pi.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class TwoVectorFrame.

        Returns:
            Scalar: At the specified times, the angle from the reference Frame's X-axis,
            along its X-Y plane, to the ascending node of this Frame's X-Y plane.

        Notes:
            A TwoVectorFrame is a fixed Frame, so its node relative to the `reference`
            Frame is independent of time. The returned Scalar always has the shape of this
            Frame, regardless of the shape of `time`.
        """

        return self._node

##########################################################################################

Frame._FRAME_SUBCLASSES.append(TwoVectorFrame)

##########################################################################################
