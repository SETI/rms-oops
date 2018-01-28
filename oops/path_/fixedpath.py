################################################################################
# oops/path_/fixedpath.py: Subclass FixedPath of class Path
################################################################################

import numpy as np
from polymath import *

from oops.event        import Event
from oops.path_.path   import Path
from oops.frame_.frame import Frame

class FixedPath(Path):
    """A path described by fixed coordinates relative to another path and frame.
    """

    PACKRAT_ARGS = ['pos', 'origin', 'frame', 'path_id']

    def __init__(self, pos, origin, frame, id=None):
        """Constructor for an FixedPath.

        Input:
            pos         a Vector3 of position vectors within the frame and
                        relative to the specified origin.
            origin      the path or ID of the reference point.
            frame       the frame or ID of the frame in which the position is
                        fixed.
            id          the name under which to register the new path; None to
                        leave the path unregistered.
        """

        # Interpret the position
        pos = Vector3.as_vector3(pos)
        pos = pos.with_deriv('t', Vector3.ZERO, 'replace')
        self.pos = pos.as_readonly()

        # Required attributes
        self.path_id = id
        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame) or self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.pos, self.origin.shape,
                                                        self.frame.shape)

        # Update waypoint and path_id; register only if necessary
        self.register()

    ########################################

    def event_at_time(self, time, quick=False):
        """Return an Event corresponding to a specified time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity on the path.
        """

        return Event(time, self.pos, self.origin, self.frame)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_FixedPath(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
