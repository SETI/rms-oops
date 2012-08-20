################################################################################
# oops_/path/fixedpath.py: Subclass FixedPath of class Path
#
# 8/19/12 MRS - changed call to register() to reregister(), so a newly defined
#   multipath replaces the old definition.
################################################################################

import numpy as np

from oops_.path.path_ import Path
from oops_.array.all import *
from oops_.event import Event

class FixedPath(Path):
    """Subclass FixedPath of class Path remains at fixed coordinates relative to
    a specified path and frame."""

    def __init__(self, pos, origin, frame, id=None):
        """Constructor for an FixedPath.

        Input:
            pos         a Vector3 of position vectors within the frame and
                        relative to the specified origin. The shape of this
                        object defines the shape of the path.
            origin      the path or ID of the center of the circle.
            frame       the frame or ID of the frame in which the circular
                        motion is defined.
            id          the name under which to register the new path; None to
                        use a temporary path ID.
        """

        if id is None:
            self.path_id = registry.temporary_path_id()
        else:
            self.path_id = id

        self.origin_id = registry.as_path_id(origin)
        self.frame_id = registry.as_frame_id(frame)

        self.pos = Vector3.as_vector3(pos)

        self.reregister()

########################################

    def event_at_time(self, time, quick=None):
        """Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        return Event(time, pos, Vector3.ZERO, self.origin_id, self.frame_id)

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
