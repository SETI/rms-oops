################################################################################
# oops_/path/multipath.py: Subclass MultiPath of class Path
################################################################################

import numpy as np
import cspice

from oops_.path.path_ import Path
from oops_.array.all import *
from oops_.event import Event

import oops_.registry as registry
import oops_.spice_support as spice

class MultiPath(Path):
    """A MultiPath gathers a set of paths into a single N-dimensional Path
    object."""

    def __init__(self, paths, origin="SSB", frame="J2000", id=None):
        """Constructor for a MultiPath Path.

        Input:
            paths           a tuple, list or ndarray of path IDs or objects.
            origin          the name or integer ID of the origin body's path.
            frame           the name or integer ID of the reference frame.
            id              the name or ID under which this MultiPath will be
                            registered. Default is the ID of the first path
                            with a "+" appended.
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        self.shape = [np.size(paths)]
        self.path_ids = np.empty(self.shape, dtype="object")
        self.paths    = np.empty(self.shape, dtype="object")
        self.dict = {}

        for (index, path) in np.ndenumerate(np.array(paths, dtype="object")):
            this_id = registry.as_path_id(path)
            this_path = registry.connect_paths(this_id, self.origin_id,
                                                        self.frame_id)
            self.path_ids[index] = this_id
            self.paths[index] = this_path
            self.dict[this_id] = this_path
            self.dict[index[0]] = this_path

        # Fill in the path_id
        if id is None:
            self.path_id = self.path_ids.ravel()[0] + "+"
        else:
            self.path_id = id

        self.register()

########################################

    def __getitem__(self, i):
        return self.dict[i]

    def __getslice__(self, i, j):
        return self.paths[i:j]

    def __len__(self): return self.paths.size

    def __str__(self):
        return ("MultiPath([" + self.path_id   + " - " +
                                self.origin_id + "]/" +
                                self.frame_id + ")")

########################################

    def event_at_time(self, time, quick=None):
        """Returns an Event object corresponding to a specified Scalar time on
        this path. The times are broadcasted across the shape of the MultiPath.

        Input:
            time        a time Scalar at which to evaluate the path.
            quick       False to disable QuickPaths; True for the default
                        options; a dictionary to override specific options.

        Return:         an Event object containing the time, position and
                        velocity of the paths.
        """

        # Broadcast to the same shape
        time = Scalar.as_scalar(time)

        masked = np.any(time.mask)
        if masked:
            (time_vals, time_mask,
            paths) = np.broadcast_arrays(time.vals, time.mask, self.paths)
            time_mask = np.array(time_mask.copy())
        else:
            (time_vals, paths) = np.broadcast_arrays(time.vals, self.paths)

        time_vals = time_vals.copy()

        # Create the event object
        pos = np.empty(time_vals.shape + (3,))
        vel = np.empty(time_vals.shape + (3,))

        for index, path in np.ndenumerate(paths):
            if masked and time_mask[index]:
                pos[index] = np.zeros(3)
                vel[index] = np.zeros(3)
            else:
                event = path.event_at_time(time_vals[index], quick)
                pos[index] = event.pos.vals
                vel[index] = event.vel.vals

        if masked:
            new_time = Scalar(time_vals, time_mask)
        else:
            new_time = Scalar(time_vals)

        return Event(new_time, pos, vel, self.origin_id, self.frame_id)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_MultiPath(unittest.TestCase):

    def runTest(self):

        from spicepath import SpicePath

        registry.initialize()

        sun   = SpicePath("SUN", "SSB")
        earth = SpicePath("EARTH", "SSB")
        moon  = SpicePath("MOON", "EARTH")

        test = MultiPath([sun,earth,moon], "SSB")

        self.assertEqual(test.path_id, "SUN+")
        self.assertEqual(test.shape, [3])

        # Single time
        event0 = test.event_at_time(0.)
        self.assertEqual(event0.shape, [3])

        # Triple of times, shape = [3]
        event012 = test.event_at_time((0., 1.e5, 2.e5))
        self.assertEqual(event012.shape, [3])

        self.assertTrue(event012.pos[0] == event0.pos[0])
        self.assertTrue(event012.vel[0] == event0.vel[0])
        self.assertTrue(event012.pos[1] != event0.pos[1])
        self.assertTrue(event012.vel[1] != event0.vel[1])
        self.assertTrue(event012.pos[2] != event0.pos[2])
        self.assertTrue(event012.vel[2] != event0.vel[2])

        # Times shaped [2,1]
        event01x = test.event_at_time([[0.], [1.e5]])
        self.assertEqual(event01x.shape, [2,3])

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
        self.assertEqual(event012a.shape, [3,3])

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

        registry.initialize()
        spice.initialize()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
