##########################################################################################
# oops/path/fixedpath.py
##########################################################################################

from polymath          import Qube, Vector3
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path


class FixedPath(Path):
    """A Path subclass describing fixed coordinates relative to another Path and
    Frame.
    """

    _WAYPOINTS = {}

    def __init__(self, pos, origin, frame=None, *, path_id=None):
        """Constructor for a FixedPath.

        Parameters:
            pos (Vector3 or array-like): The position vectors within the frame and
                relative to the specified origin.
            origin (Path or str): The Path or the ID of the Path defining the origin.
            frame (Frame or str, optional): The Frame or the ID of the Frame in which the
                fixed coordinates are defined and in which they are returned; None to use
                the Frame of the `origin` Path.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            KeyError: If `origin` or `frame` is an ID string that has not been registered.
            ValueError: If the shapes of `pos`, `origin`, and `frame` cannot be
                broadcasted.
        """

        # Interpret the position
        pos = Vector3.as_vector3(pos)
        pos = pos.with_deriv('t', Vector3.ZERO, method='replace')
        self._pos = pos.as_readonly()

        # Required attributes
        self._origin = Path.as_waypoint(origin)
        self._frame = (frame and Frame.as_wayframe(frame)) or self._origin._frame
        self._shape = Qube.broadcasted_shape(self._pos, self._origin._shape,
                                             self._frame._shape)

        self._register(path_id)
        self.refresh()

    def _waypoint_key(self):
        return (self._pos, self._origin, self._frame)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._pos, self._origin, self._frame, self.stripped_id)

    def __setstate__(self, state):
        (pos, origin, frame, path_id) = state
        self.__init__(pos, origin, frame, path_id=path_id)
        self.freeze()

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        return Event(time, self._pos, self._origin, self._frame)

##########################################################################################

Path._PATH_SUBCLASSES.append(FixedPath)

##########################################################################################
