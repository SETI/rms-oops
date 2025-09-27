##########################################################################################
# oops/path/multipath.py: Subclass MultiPath of class Path
##########################################################################################

import numpy as np

from polymath          import Qube, Scalar
from oops.event        import Event
from oops.fittable     import Fittable_
from oops.frame.frame_ import Frame
from oops.path.path_   import Path


class MultiPath(Path):
    """Gathers a set of paths into a single 1-D Path object."""

    _PATH_IDS = {}

    def __init__(self, paths, origin=None, frame=None, path_id=None):
        """Constructor for a MultiPath Path.

        Parameters:
            paths (tuple or list): Paths or path IDs to include in this MultiPath.
            origin (Path or str, optional): Path or ID identifying the common origin of
                all paths. None to use the SSB.
            frame (Frame or str, optional): Frame or ID identifying the reference frame.
                None to use the default frame of the `origin` path.
            path_id (str, optional): The ID to use; None to leave the path unregistered.
                Use '+' for the names of all the paths appended with "+".
        """

        # Interpret the inputs
        self.origin = Path.as_waypoint(origin) or Path.SSB
        self.frame = Frame.as_wayframe(frame) or self.origin.frame

        self.paths = np.array(paths, dtype='object').ravel()
        self.shape = self.paths.shape

        for (index, path) in np.ndenumerate(self.paths):
            self.paths[index] = Path.as_path(path).wrt(self.origin, self.frame)

        # Fill in the path_id
        if path_id == '+':
            if len(self.paths) == 1:
                self.path_id = self.paths[0].path_id + '+'
            else:
                self.path_id = '+'.join(p.path_id for p in self.paths)
        else:
            self.path_id = self._recover_id(path_id)

        self.register()
        self._cache_id()

    # Support indexing by integer and numeric range
    def __getitem__(self, i):
        paths = self.paths[i]
        if np.shape(paths) == ():
            return paths
        return MultiPath(paths, self.origin, self.frame, path_id=None)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return list(self.paths) + [self.origin, self.frame]

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.paths, Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame), self._state_id())

    def __setstate__(self, state):
        (paths, origin, frame, path_id) = state
        self.__init__(paths, origin, frame, path_id=path_id)
        Fittable_.freeze(self)

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, quick={}):
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

        # Broadcast everything to the same shape
        time = Qube.broadcast(Scalar.as_scalar(time), self.shape)[0]

        # Create the event object
        pos = np.empty(time.shape + (3,))
        vel = np.empty(time.shape + (3,))
        mask = np.empty(time.shape, dtype='bool')
        mask[...] = time.mask

        for (index, path) in np.ndenumerate(self.paths):
            event = path.event_at_time(time.values[..., index], quick=quick)
            pos[..., index, :] = event.pos.values
            vel[..., index, :] = event.vel.values
            mask[..., index] |= (event.pos.mask | event.vel.mask)

        if not np.any(mask):
            mask = False
        elif np.all(mask):
            mask = True

        return Event(Scalar(time.values, mask), (pos,vel), self.origin, self.frame)

    def quick_path(self, time, quick={}):
        """Override of the default quick_path method to return a MultiPath of quick_paths.

        A QuickPath operates by sampling the given path and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed up
        performance when the same path must be evaluated many times, e.g., for every pixel
        of an image.

        Parameters:
            time (Scalar or array-like): The times at which the frame is to be evaluated.
                Alternatively, a tuple (minimum time, maximum time, number of times)
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters;
                use False to disable the use of QuickPaths and QuickFrames.
            quick (dict or bool, optional): If False, no QuickPath is created and self is
                returned. If a dictionary is given, its values override the values in the
                default dictionary QUICK.dictionary and the merged dictionary is used.
        """

        new_paths = []
        for path in self.paths:
            new_path = path.quick_path(time, quick=quick)
            new_paths.append(new_path)

        return MultiPath(new_paths, self.origin, self.frame)

##########################################################################################
