##########################################################################################
# oops/path/circlepath.py: Subclass CirclePath of class Path
##########################################################################################

from polymath          import Qube, Scalar, Vector3
from oops.cache        import Cache
from oops.event        import Event
from oops.fittable     import Fittable_
from oops.frame.frame_ import Frame
from oops.path.path_   import Path


class CirclePath(Path):
    """A path describing uniform circular motion about another path.

    The orientation of the circle is defined by the z-axis of the given frame.
    """

    _PATH_IDS = {}

    def __init__(self, radius, lon, rate, epoch, origin, frame=None, path_id=None):
        """Constructor for a CirclePath.

        Parameters:
            radius (Scalar, array-like, or float): Radius of the path, km.
            lon (Scalar, array-like, or float): Longitude of the path at epoch, measured
                from the x-axis of the frame, toward the y-axis, in radians.
            rate (Scalar, array-like, or float): Rate of circular motion, radians/second.
            epoch (Scalar, array-like, or float): The time TDB relative to which all
                orbital elements are defined.
            origin (Path or str): The path or ID of the center of the circle.
            frame (Frame or str): The frame or ID of the frame in which the circular
                motion is defined; None to use the default frame of the origin path.
            path_id (str, optional): The ID to use; None to leave the path unregistered.

        Notes:
            The shape of the Path object returned is defined by broadcasting together the
            shapes of all the orbital elements plus the epoch.
        """

        # Interpret the elements
        self.epoch  = Scalar.as_scalar(epoch)
        self.radius = Scalar.as_scalar(radius)
        self.lon    = Scalar.as_scalar(lon)
        self.rate   = Scalar.as_scalar(rate)

        # Required attributes
        self.origin = Path.as_waypoint(origin)
        self.frame = Frame.as_wayframe(frame) or self.origin.frame
        self.shape = Qube.broadcasted_shape(self.radius, self.lon, self.rate,
                                            self.epoch, self.origin.shape,
                                            self.frame.shape)
        self.path_id = self._recover_id(path_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return (self.radius, self.lon, self.rate, self.epoch, self.origin, self.frame)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.radius, self.lon, self.rate, self.epoch,
                Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame), self._state_id())

    def __setstate__(self, state):
        (radius, lon, rate, epoch, origin, frame, path_id) = state
        self.__init__(radius, lon, rate, epoch, origin, frame, path_id=path_id)
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

        lon = self.lon + self.rate * (Scalar.as_scalar(time) - self.epoch)
        r_cos_lon = self.radius * lon.cos()
        r_sin_lon = self.radius * lon.sin()

        pos = Vector3.from_scalars(r_cos_lon, r_sin_lon, 0.)
        vel = Vector3.from_scalars(-r_sin_lon * self.rate, r_cos_lon * self.rate, 0.)

        return Event(time, (pos,vel), self.origin, self.frame)

##########################################################################################
