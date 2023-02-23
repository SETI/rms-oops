################################################################################
# oops/path/multipath.py: Subclass MultiPath of class Path
################################################################################

import numpy as np
from polymath import Qube, Scalar

from .       import Path
from ..event import Event
from ..frame import Frame

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
        return (self.paths, self.origin, self.frame)

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
# UNIT TESTS
################################################################################

import unittest

class Test_MultiPath(unittest.TestCase):

    def runTest(self):

        import cspyce
        from .spicepath import SpicePath
        from ..unittester_support import TESTDATA_PARENT_DIRECTORY
        import os

        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "naif0009.tls"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "pck00010.tpc"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "de421.bsp"))

        Path.reset_registry()
        Frame.reset_registry()

        sun   = SpicePath("SUN", "SSB")
        earth = SpicePath("EARTH", "SSB")
        moon  = SpicePath("MOON", "EARTH")

        test = MultiPath([sun,earth,moon], "SSB")

        self.assertEqual(test.path_id, "SUN+others")
        self.assertEqual(test.shape, (3,))

        # Single time
        event0 = test.event_at_time(0.)
        self.assertEqual(event0.shape, (3,))

        # Triple of times, shape = [3]
        event012 = test.event_at_time((0., 1.e5, 2.e5))
        self.assertEqual(event012.shape, (3,))

        self.assertTrue(event012.pos[0] == event0.pos[0])
        self.assertTrue(event012.vel[0] == event0.vel[0])
        self.assertTrue(event012.pos[1] != event0.pos[1])
        self.assertTrue(event012.vel[1] != event0.vel[1])
        self.assertTrue(event012.pos[2] != event0.pos[2])
        self.assertTrue(event012.vel[2] != event0.vel[2])

        # Times shaped [2,1]
        event01x = test.event_at_time([[0.], [1.e5]])
        self.assertEqual(event01x.shape, (2,3))

        self.assertTrue(event01x.pos[0,0] == event0.pos[0])
        self.assertTrue(event01x.vel[0,0] == event0.vel[0])
        self.assertTrue(event01x.pos[0,1] == event0.pos[1])
        self.assertTrue(event01x.vel[0,1] == event0.vel[1])
        self.assertTrue(event01x.pos[0,2] == event0.pos[2])
        self.assertTrue(event01x.vel[0,2] == event0.vel[2])

        self.assertTrue(event01x.pos[1,1] == event012.pos[1])
        self.assertTrue(event01x.vel[1,1] == event012.vel[1])
        self.assertTrue(event01x.pos[1,2] != event012.pos[2])
        self.assertTrue(event01x.vel[1,2] != event012.pos[2])

        # Triple of times, at all times, shape [3,1]
        event012a = test.event_at_time([[0.], [1.e5], [2.e5]])
        self.assertEqual(event012a.shape, (3,3))

        self.assertTrue(event012a.pos[0,:] == event0.pos)
        self.assertTrue(event012a.vel[0,:] == event0.vel)

        self.assertTrue(event012a.pos[0,0] == event012.pos[0])
        self.assertTrue(event012a.vel[0,0] == event012.vel[0])
        self.assertTrue(event012a.pos[1,1] == event012.pos[1])
        self.assertTrue(event012a.vel[1,1] == event012.vel[1])
        self.assertTrue(event012a.pos[2,2] == event012.pos[2])
        self.assertTrue(event012a.vel[2,2] == event012.vel[2])

        self.assertTrue(event012a.pos[0:2] == event01x.pos)
        self.assertTrue(event012a.vel[0:2] == event01x.vel)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
