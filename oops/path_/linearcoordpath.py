################################################################################
# oops_/path/linearcoordpath.py: Subclass LinearCoordPath of class Path
################################################################################

import numpy as np
from polymath import *

from oops.event        import Event
from oops.path_.path   import Path
from oops.frame_.frame import Frame

class LinearCoordPath(Path):
    """A path defined by coordinates changing linearly on a specified Surface.
    """

    PACKRAT_ARGS = ['surface', 'coords', 'coords_dot', 'epoch', 'obs_path',
                    'path_id']

    def __init__(self, surface, coords, coords_dot, epoch, obs=None, id=None):
        """Constructor for a CoordPath.

        Input:
            surface     a surface.
            coords      a tuple of 2 or 3 Scalars defining the coordinates on
                        the surface.
            coords_dot  the time-derivative of the coords.
            epoch       the epoch at which the coords are defined, seconds TDB.
            obs         optional path of observer, needed to calculate points
                        on virtual surfaces.
            id          the name under which to register the new path; None to
                        leave the path unregistered.
        """

        self.surface = surface
        self.coords = coords
        self.coords_dot = coords_dot
        self.epoch  = epoch
        self.obs_path = obs

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

        new_coords = []
        for i in range(len(self.coords)):
            coord = self.coords[i] + self.coords_dot[i] * (time - self.epoch)
            coord.insert_deriv('t', self.coords_dot[i])
            new_coords.append(coord)

        new_coords = tuple(new_coords)

        if self.surface.IS_VIRTUAL:
            obs = self.obs_path.event_at_time(time, quick=quick).pos
        else:
            obs = None

        pos = self.surface.vector3_from_coords(new_coords, obs, derivs=True)

        return Event(time, pos, self.origin, self.frame)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_LinearCoordPath(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
