##########################################################################################
# oops/path/circlepath.py: Subclass CirclePath of class Path
##########################################################################################

from polymath          import Qube, Scalar, Vector3
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path
import oops.mutable as mutable


class CirclePath(Path):
    """A path describing uniform circular motion about another path.

    The orientation of the circle is defined by the z-axis of the given frame.
    """

    _WAYPOINTS = {}

    def __init__(self, radius, lon, rate, epoch, origin, frame=None, *, path_id=None):
        """Constructor for a CirclePath.

        Parameters:
            radius (Scalar, array-like, or float): Radius of the path, km.
            lon (Scalar, array-like, or float): Longitude of the path at epoch, measured
                from the x-axis of the frame, toward the y-axis, in radians.
            rate (Scalar, array-like, or float): Rate of circular motion, radians/second.
            epoch (Scalar, array-like, or float): The time TDB relative to which all
                orbital elements are defined.
            origin (Path or str): The path or ID of the center of the circle.
            frame (Frame or str, optional): The Frame or the ID of the Frame in which the
                circular motion is defineed and in which coordinates are returned; None to
                use the frame of the `origin` path.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            ValueError: If the shapes of `radius`, `lon`, `rate`, `epoch`, `origin`, and
                `frame` cannot be broadcasted.
        """

        # Interpret the elements
        self._radius = Scalar.as_scalar(radius).as_readonly()
        self._lon    = Scalar.as_scalar(lon).as_readonly()
        self._rate   = Scalar.as_scalar(rate).as_readonly()
        self._epoch  = Scalar.as_scalar(epoch).as_readonly()

        self._origin = Path.as_waypoint(origin)
        self._frame = (frame and Frame.as_wayframe(frame)) or self._origin._frame
        self._shape = Qube.broadcasted_shape(self._radius, self._lon, self._rate,
                                             self._epoch, self._origin._shape,
                                             self._frame._shape)

        self._register(path_id)
        mutable.refresh(self)

    def _waypoint_key(self):
        return (self._radius, self._lon, self._rate, self._epoch, self._origin,
                self._frame)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._radius, self._lon, self._rate, self._epoch, self._origin,
                self._frame, self.stripped_id)

    def __setstate__(self, state):
        (radius, lon, rate, epoch, origin, frame, path_id) = state
        self.__init__(radius, lon, rate, epoch, origin, frame=frame, path_id=path_id)
        mutable.freeze(self)

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Event): The Event object containing (at least) the time, position, and
                velocity on the Path.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        lon = self._lon + self._rate * (Scalar.as_scalar(time) - self._epoch)
        r_cos_lon = self._radius * lon.cos()
        r_sin_lon = self._radius * lon.sin()

        pos = Vector3.from_scalars(r_cos_lon, r_sin_lon, 0.)
        vel = Vector3.from_scalars(-r_sin_lon * self._rate, r_cos_lon * self._rate, 0.)

        return Event(time, (pos, vel), self._origin, self._frame)

##########################################################################################

Path._PATH_SUBCLASSES.append(CirclePath)

##########################################################################################
