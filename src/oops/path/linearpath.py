##########################################################################################
# oops/path/linearpath.py
##########################################################################################

from polymath          import Qube, Scalar, Vector3
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path


class LinearPath(Path):
    """A Path subclass describing linear motion relative to another Path and Frame."""

    _WAYPOINTS = {}

    def __init__(self, pos, epoch, origin, *, frame=None, path_id=None):
        """Constructor for a LinearPath.

        Parameters:
            pos (Vector3, array-like, or tuple): Position vector(s). The velocity is
                defined via a derivative 'd_dt'. Alternatively, provide (pos, vel) as a
                tuple of two Vector3 or array-like values.
            epoch (Scalar, array-like, or float): The time TDB relative to which all
                orbital elements are defined.
            origin (Path or str): The Path or the ID of the Path defining the origin of
                the linear path.
            frame (Frame or str, optional): The Frame or the ID of the Frame in which the
                linear motion is expressed and coordinates are returned; by default, this
                is the Frame of the `origin` Path.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            KeyError: If `origin` or `frame` is an ID string that has not been registered.
            ValueError: If the shapes of `pos`, `epoch`, `origin`, and `frame` cannot be
                broadcasted.
        """

        # Interpret the position
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            self._pos = Vector3.as_vector3(pos[0]).wod.as_readonly()
            self._vel = Vector3.as_vector3(pos[1]).wod.as_readonly()
        else:
            pos = Vector3.as_vector3(pos)
            self._vel = (pos.d_dt.wod.as_readonly() if hasattr(pos, 'd_dt')
                         else Vector3.ZERO)
            self._pos = pos.wod.as_readonly()

        self._epoch = Scalar.as_scalar(epoch)

        # Required attributes
        self._origin = Path.as_waypoint(origin)
        self._frame = frame and Frame.as_wayframe(frame) or self._origin._frame
        self._shape = Qube.broadcasted_shape(self._pos, self._epoch, self._origin._shape,
                                             self._frame._shape)

        self._register(path_id)
        self.refresh()

    def _waypoint_key(self):
        return (self._pos, self._vel, self._epoch, self._origin, self._frame)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        parts = [f'{name}(pos = ({self._pos.mvals}',
                 f'{blanks}        {self._vel.mvals})',
                 f'{blanks}epoch = {self._epoch}',
                 f'{blanks}origin = {self._origin.show(level-1, skip+9)}']
        if self._frame != self._origin._frame:
            parts.append(f'{blanks}frame = {self._frame.show(level-1, skip+8)}')

        return ',\n'.join(parts) + ')'

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._pos, self._vel, self._epoch, self._origin, self._frame,
                self.stripped_id)

    def __setstate__(self, state):
        (pos, vel, epoch, origin, frame, path_id) = state
        self.__init__((pos, vel), epoch, origin, frame=frame, path_id=path_id)
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

        return Event(time, (self._pos + (time - self._epoch) * self._vel, self._vel),
                     self._origin, self._frame)

##########################################################################################

Path._PATH_SUBCLASSES.append(LinearPath)

##########################################################################################
