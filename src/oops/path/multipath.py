##########################################################################################
# oops/path/multipath.py
##########################################################################################

import numpy as np

from polymath          import Qube, Scalar, Vector3
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path


class MultiPath(Path):
    """A Path subclass that gathers a set of Paths into a single 1-D Path."""

    _WAYPOINTS = {}

    def __init__(self, paths, origin=None, frame=None, *, path_id=None):
        """Constructor for a MultiPath.

        Parameters:
            paths (list[Path or str]): The Paths or Path IDs to include in this MultiPath.
            origin (Path or str, optional): The Path or the ID of the Path defining the
                common origin of all the Paths; None to use the SSB.
            frame (Frame or str, optional): The Frame or the ID of the Frame in which
                coordinates are returned; None to use the Frame of the `origin` Path.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            KeyError: If `paths`, `origin`, or `frame` contains an ID string that has not
                been registered.
            ValueError: If `paths` is not one-dimensional.
        """

        # Interpret the inputs
        self._origin = origin and Path.as_waypoint(origin) or Path.SSB
        self._frame = frame and Frame.as_wayframe(frame) or self._origin._frame

        self._input_paths = np.array(paths, dtype='object')
        if self._input_paths.ndim != 1:
            raise ValueError('a MultiPath cannot be multidimensional')

        for k, path in enumerate(self._input_paths):
            self._input_paths[k] = Path.as_path(path)

        self._shape = self._input_paths.shape
        self._paths = np.empty(self._shape, dtype='object')

        # The waypoint key is derived from self._paths, which refresh() fills in, so the
        # paths have to be resolved before this object is registered. Registering first
        # would give every MultiPath of the same length the same key, and each one would
        # then share the waypoint of the first one created.
        self.refresh()
        self._register(path_id)

    def _refresh(self):
        for k, path in np.ndenumerate(self._input_paths):
            self._paths[k] = path.wrt(self._origin, self._frame)

    def __getitem__(self, i):
        """The Path or MultiPath selected by an index or a numeric range."""
        paths = self._paths[i]
        if np.shape(paths) == ():
            return paths
        return MultiPath(paths, self._origin, self._frame, path_id=None)

    def _waypoint_key(self):
        return self._paths

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        parts = [f'{name}(paths = ({self._paths[0].show(level-1, skip+9)}']
        for path in self._paths[1:-1]:
            parts.append(f'{blanks}         {path.show(level-1, skip+9)}')
        parts.append(f'{blanks}         {self._paths[-1].show(level-1, skip+9)})')

        parts.append(f'{blanks}origin = {self._origin.show(level-1, skip+9)}')
        if self._frame != self._origin._frame:
            parts.append(f'{blanks}frame = {self._frame.show(level-1, skip+8)}')

        return ',\n'.join(parts) + ')'

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._paths, self._origin, self._frame, self.stripped_id)

    def __setstate__(self, state):
        (paths, origin, frame, path_id) = state
        self.__init__(paths, origin, frame, path_id=path_id)
        self.freeze()

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default :class:`~oops.path.QuickPath` and
                :class:`~oops.frame.QuickFrame` parameters. Use False to disable the use
                of QuickPaths and QuickFrames. The default quick dictionary is defined in
                config.py.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        # Broadcast everything to the same shape
        time = Qube.broadcast(Scalar.as_scalar(time), self._shape)[0]
        time = time.expand_mask()

        # Create the event object
        pos = np.empty(time.shape + (3,))
        vel = np.empty(time.shape + (3,))
        mask = np.empty(time.shape, dtype=bool)

        for (k, path) in enumerate(self._paths):
            if np.all(time.mask[..., k]):
                pos[..., k, :] = 1.
                vel[..., k, :] = 1.
                mask[..., k] = True
            else:
                event = path.event_at_time(time.vals[..., k], quick=quick)
                pos[..., k, :] = event.pos.vals
                vel[..., k, :] = event.vel.vals
                mask[..., k] = event.pos.mask

        return Event(Scalar(time.vals, mask), (Vector3(pos, mask), Vector3(vel, mask)),
                     self._origin, self._frame)

    def quick_path(self, time, *, quick=None):
        """A MultiPath of the QuickPaths that approximate this MultiPath's Paths.

        A :class:`~oops.path.QuickPath` operates by sampling the given Path and then
        setting up an interpolation grid to evaluate in its place. It can substantially
        speed up performance when the same Path must be evaluated many times, e.g., for
        every pixel of an image.

        Parameters:
            time (Scalar): The time(s) at which this path is to be evaluated.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default :class:`~oops.path.QuickPath` and
                :class:`~oops.frame.QuickFrame` parameters. Use False to disable the use
                of QuickPaths and QuickFrames. The default quick dictionary is defined in
                config.py.

        Returns:
            MultiPath: A MultiPath of the QuickPaths that approximate the individual Paths
            for the given range of times.
        """

        if quick == False:  # noqa E712 -- allow bool, np.bool_, and int 0, but not None
            return self

        # Broadcast everything to the same shape
        time = Qube.broadcast(Scalar.as_scalar(time), self._shape)[0]

        new_paths = np.empty(self._shape, dtype='object')
        for k, path in enumerate(self._input_paths):
            new_paths[k] = path.quick_path(time[..., k], quick=quick)

        return MultiPath(new_paths, self._origin, self._frame)

##########################################################################################

Path._PATH_SUBCLASSES.append(MultiPath)

##########################################################################################
