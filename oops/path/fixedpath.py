##########################################################################################
# oops/path/fixedpath.py: Subclass FixedPath of class Path
##########################################################################################

from polymath          import Qube, Vector3
from oops.event        import Event
from oops.fittable     import Fittable_
from oops.frame.frame_ import Frame
from oops.path.path_   import Path


class FixedPath(Path):
    """A path described by fixed coordinates relative to another path and frame."""

    _PATH_IDS = {}

    def __init__(self, pos, origin, frame, path_id=None):
        """Constructor for an FixedPath.

        Parameters:
            pos (Vector3 or array-like): The position vectors within the frame and
                relative to the specified origin.
            origin (Path or str): The path or ID of the center of the circle.
            frame (Frame or str): The frame or ID of the frame in which the fixed
                coordinates are defined.
            path_id (str, optional): The ID to use; None to leave the path unregistered.
        """

        # Interpret the position
        pos = Vector3.as_vector3(pos)
        pos = pos.with_deriv('t', Vector3.ZERO, method='replace')
        self.pos = pos.as_readonly()

        # Required attributes
        self.origin = Path.as_waypoint(origin)
        self.frame = Frame.as_wayframe(frame) or self.origin.frame
        self.shape = Qube.broadcasted_shape(self.pos, self.origin, self.frame)
        self.path_id = Path._recover_id(path_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return (self.pos, self.origin, self.frame)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.pos, Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame), self._state_id())

    def __setstate__(self, state):
        (pos, origin, frame, path_id) = state
        self.__init__(pos, origin, frame, path_id=path_id)
        Fittable_.freeze(self)

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, quick=False):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar, array-like, or float): Time at which to evaluate the path, in
                seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters;
                use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Event): Event object containing (at least) the time, position, and velocity
                on the path.
        """

        return Event(time, self.pos, self.origin, self.frame)

##########################################################################################
