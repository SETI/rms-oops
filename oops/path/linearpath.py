##########################################################################################
# oops/path/linearpath.py: Subclass LinearPath of class Path
##########################################################################################

from polymath          import Qube, Scalar, Vector3
from oops.event        import Event
from oops.fittable     import Fittable_
from oops.frame.frame_ import Frame
from oops.path.path_   import Path

class LinearPath(Path):
    """A path defining linear motion relative to another path and frame."""

    _PATH_IDS = {}

    def __init__(self, pos, epoch, origin, frame=None, path_id=None):
        """Constructor for a LinearPath.

        Input:
            pos (Vector3, array-like, or tuple): Position vector(s). The velocity is
                defined via a derivative 'd_dt'. Alternatively, provide (pos, vel) as a
                tuple of two Vector3 or array-like values.
            epoch (Scalar, array-like, or float): The time TDB relative to which all
                orbital elements are defined.
            origin (Path or str): The path or ID of the center of the circle.
            frame (Frame or str): The frame or ID of the frame in which the fixed
                coordinates are defined.
            path_id (str, optional): The ID to use; None to leave the path unregistered.
        """

        # Interpret the position
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            self.pos = Vector3.as_vector3(pos[0]).wod.as_readonly()
            self.vel = Vector3.as_vector3(pos[1]).wod.as_readonly()
        else:
            pos = Vector3.as_vector3(pos)

            if hasattr(pos, 'd_dt'):
                self.vel = pos.d_dt.as_readonly()
            else:
                self.vel = Vector3.ZERO

            self.pos = pos.wod.as_readonly()

        self.epoch = Scalar.as_scalar(epoch)

        # Required attributes
        self.origin = Path.as_waypoint(origin)
        self.frame = Frame.as_wayframe(frame) or self.origin.frame
        self.shape = Qube.broadcasted_shape(self.pos, self.vel, self.epoch, self.origin,
                                            self.frame)
        self.path_id = self._recover_id(path_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return (self.pos, self.vel, self.epoch, self.origin, self.frame)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.pos, self.vel, self.epoch, Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame), self._state_id())

    def __setstate__(self, state):
        (pos, vel, epoch, origin, frame, path_id) = state
        self.__init__((pos, vel), epoch, origin, frame, path_id=path_id)
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

        return Event(time, (self.pos + (time-self.epoch) * self.vel, self.vel),
                           self.origin, self.frame)

############################################################################################
