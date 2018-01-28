################################################################################
# oops_/path/coordpath.py: Subclass CoordPath of class Path
################################################################################

import numpy as np
from polymath import *

from oops.event        import Event
from oops.path_.path   import Path
from oops.frame_.frame import Frame

class CoordPath(Path):
    """A path defined by fixed coordinates on a specified Surface."""

    PACKRAT_ARGS = ['surface', 'coords', 'obs_path', 'path_id']

    def __init__(self, surface, coords, obs=None, id=None):
        """Constructor for a CoordPath.

        Input:
            surface     a surface.
            coords      a tuple of 2 or 3 Scalars defining the coordinates on
                        the surface.
            obs         optional path of observer, needed to calculate points
                        on virtual surfaces.
            id          the name under which to register the new path; None to
                        leave the path unregistered.
        """

        self.surface = surface
        self.coords = coords
        self.obs_path = obs

        if not self.surface.IS_VIRTUAL:
            self.pos = self.surface.vector3_from_coords(self.coords)
        else:
            self.pos = None

        # Required attributes
        self.path_id = id
        self.origin  = self.surface.origin
        self.frame   = self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(*coords)

        # Update waypoint and path_id; register only if necessary
        self.register()

    ########################################

    def event_at_time(self, time, quick={}):
        """Return an Event corresponding to a specified time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity on the path.
        """

        if self.surface.IS_VIRTUAL:
            obs_event = self.obs_path.event_at_time(time, quick=quick)
            self.pos = self.surface.vector3_from_coords(self.coords,
                                                        obs_event.pos)

        return Event(time, self.pos, self.origin, self.frame)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_CoordPath(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
