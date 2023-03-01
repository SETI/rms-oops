################################################################################
# oops/path/fixedpath.py: Subclass FixedPath of class Path
################################################################################

from polymath import Qube, Vector3

from oops.event import Event
from oops.frame import Frame
from oops.path  import Path

class FixedPath(Path):
    """A path described by fixed coordinates relative to another path and frame.
    """

    # Note: FixedPaths are not generally re-used, so their IDs are expendable.
    # Their IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, pos, origin, frame, path_id=None):
        """Constructor for an FixedPath.

        Input:
            pos         a Vector3 of position vectors within the frame and
                        relative to the specified origin.
            origin      the path or ID of the reference point.
            frame       the frame or ID of the frame in which the position is
                        fixed.
            path_id     the name under which to register the new path; None to
                        leave the path unregistered.
        """

        # Interpret the position
        pos = Vector3.as_vector3(pos)
        pos = pos.with_deriv('t', Vector3.ZERO, 'replace')
        self.pos = pos.as_readonly()

        # Required attributes
        self.path_id = path_id
        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame) or self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.pos, self.origin, self.frame)

        # Update waypoint and path_id; register only if necessary
        self.register()

    # Unpickled paths will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.pos,
                Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame))

    def __setstate__(self, state):
        self.__init__(*state)

    #==========================================================================
    def event_at_time(self, time, quick=False):
        """An Event corresponding to a specified time on this path.

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
