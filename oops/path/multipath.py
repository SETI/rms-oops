################################################################################
# oops/path/multipath.py: Subclass MultiPath of class Path
################################################################################

import numpy as np

from polymath          import Qube, Scalar
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path

class MultiPath(Path):
    """Gathers a set of paths into a single 1-D Path object."""

    PATH_IDS = {}

    #===========================================================================
    def __init__(self, paths, origin=None, frame=None, path_id='+',
                       unpickled=False):
        """Constructor for a MultiPath Path.

        Input:
            paths       a tuple, list or 1-D ndarray of paths or path IDs.
            origin      a path or path ID identifying the common origin of all
                        paths. None to use the SSB.
            frame       a frame or frame ID identifying the reference frame.
                        None to use the default frame of the origin path.
            path_id     the name or ID under which this path will be registered.
                        A single '+' is changed to the ID of the first path with
                        a '+' appended. None to leave the path unregistered.
            unpickled   True if this path has been read from a pickle file.
        """

        # Interpret the inputs
        self.origin = Path.as_waypoint(origin) or Path.SSB
        self.frame  = Frame.as_wayframe(frame) or self.origin.frame

        self.paths = np.array(paths, dtype='object').ravel()
        self.shape = self.paths.shape
        self.keys = set()

        for (index, path) in np.ndenumerate(self.paths):
            self.paths[index] = Path.as_path(path).wrt(self.origin, self.frame)

        # Fill in the path_id
        self.path_id = path_id

        if self.path_id == '+':
            self.path_id = self.paths[0].path_id + '+others'

        # Update waypoint and path_id; register only if necessary
        self.register(unpickled=unpickled)

        # Save in internal dict for name lookup upon serialization
        if not unpickled and self.path_id in Path.WAYPOINT_REGISTRY:
            key = tuple([path.path_id for path in self.paths])
            MultiPath.PATH_IDS[key] = self.path_id

    # Unpickled paths will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.paths,
                Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame))

    def __setstate__(self, state):
        # If this path matches a pre-existing path, re-use its ID
        (paths, origin, frame) = state
        key = tuple([path.path_id for path in paths])
        path_id = MultiPath.PATH_IDS.get(key, None)
        self.__init__(paths, origin, frame, path_id=path_id, unpickled=True)

    #===========================================================================
    def __getitem__(self, i):
        slice = self.paths[i]
        if np.shape(slice) == ():
            return slice
        return MultiPath(slice, self.origin, self.frame, path_id=None)

    #===========================================================================
    def event_at_time(self, time, quick={}):
        """An Event object corresponding to a specified Scalar time on this
        path.

        The times are broadcasted across the shape of the MultiPath.

        Input:
            time        a time Scalar at which to evaluate the path.
            quick       False to disable QuickPaths; a dictionary to override
                        specific options.

        Return:         an Event object containing the time, position and
                        velocity of the paths.
        """

        # Broadcast everything to the same shape
        time = Qube.broadcast(Scalar.as_scalar(time), self.shape)[0]

        # Create the event object
        pos = np.empty(time.shape + (3,))
        vel = np.empty(time.shape + (3,))
        mask = np.empty(time.shape, dtype='bool')
        mask[...] = time.mask

        for (index, path) in np.ndenumerate(self.paths):
            event = path.event_at_time(time.values[...,index], quick=quick)
            pos[...,index,:] = event.pos.values
            vel[...,index,:] = event.vel.values
            mask[...,index] |= (event.pos.mask | event.vel.mask)

        if not np.any(mask):
            mask = False
        elif np.all(mask):
            mask = True

        return Event(Scalar(time.values, mask), (pos,vel),
                            self.origin, self.frame)

    #===========================================================================
    def quick_path(self, time, quick={}):
        """Override of the default quick_path method to return a MultiPath of
        quick_paths.

        A QuickPath operates by sampling the given path and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed
        up performance when the same path must be evaluated many times, e.g.,
        for every pixel of an image.

        Input:
            time        a Scalar defining the set of times at which the frame is
                        to be evaluated. Alternatively, a tuple (minimum time,
                        maximum time, number of times)
            quick       if None or False, no QuickPath is created and self is
                        returned; if another dictionary, then the values
                        provided override the values in the default dictionary
                        QUICK.dictionary, and the merged dictionary is used.
        """

        new_paths = []
        for path in self.paths:
            new_path = path.quick_path(time, quick=quick)
            new_paths.append(new_path)

        return MultiPath(new_paths, self.origin, self.frame)

################################################################################
