################################################################################
# oops/path/coordpath.py: Subclass CoordPath of class Path
################################################################################

from polymath import Qube, Scalar

from .         import Path
from ..event   import Event
from ..surface import Surface

class CoordPath(Path):
    """A path defined by fixed coordinates on a specified Surface."""

    # Note: CoordPaths are not generally re-used, so their IDs are expendable.
    # Their IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, surface, coords, obs=None, path_id=None):
        """Constructor for a CoordPath.

        Input:
            surface     a surface.
            coords      a tuple of 2 or 3 Scalars defining the coordinates on
                        the surface.
            obs         optional path of observer, needed to calculate points
                        on virtual surfaces.
            path_id     the name under which to register the new path; None to
                        leave the path unregistered.
        """

        self.surface = surface
        self.coords = [Scalar(x) for x in coords]
        self.obs_path = None if obs is None else Path.as_path(obs)

        assert isinstance(self.surface, Surface)

        if not self.surface.IS_VIRTUAL:
            self.pos = self.surface.vector3_from_coords(self.coords)
        else:
            self.pos = None

        # Required attributes
        self.path_id = path_id
        self.origin  = self.surface.origin
        self.frame   = self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.surface, self.obs_path,
                                              *self.coords)

        # Update waypoint and path_id; register only if necessary
        self.register()

    # Unpickled paths will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.surface, self.coords, self.obs_path)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def event_at_time(self, time, quick={}):
        """An Event corresponding to a specified time on this path.

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
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
