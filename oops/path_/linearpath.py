################################################################################
# oops_/path/linearpath.py: Subclass LinearPath of class Path
################################################################################

import numpy as np
from polymath import *

from oops.event        import Event
from oops.path_.path   import Path
from oops.frame_.frame import Frame

class LinearPath(Path):
    """A path defining linear motion relative to another path and frame."""

    PACKRAT_ARGS = ['pos', 'epoch', 'origin', 'frame', 'path_id']

    def __init__(self, pos, epoch, origin, frame=None, id=None):
        """Constructor for a LinearPath.

        Input:
            pos         a Vector3 of position vectors. The velocity should be
                        defined via a derivative 'd_dt'. Alternatively, it can
                        be specified as a tuple of two Vector3 objects,
                        (position, velocity).
            epoch       time Scalar relative to which the motion is defined,
                        seconds TDB
            origin      the path or path ID of the reference point.
            frame       the frame or frame ID of the coordinate system; None for
                        the frame used by the origin path.
            id          the name under which to register the new path; None to
                        leave the path unregistered.
        """

        # Interpret the position
        if type(pos) in (tuple,list) and len(pos) == 2:
            self.pos = Vector3.as_vector3(pos[0]).wod.as_readonly()
            self.vel = Vector3.as_vector3(pos[1]).wod.as_readonly()
        else:
            pos = Vector3.as_vector3(pos)

            if hasattr('d_dt', pos):
                self.vel = pos.d_dt.as_readonly()
            else:
                self.vel = Vector3.ZERO

            self.pos = pos.wod.as_readonly()

        self.epoch = Scalar.as_scalar(epoch)

        # Required attributes
        self.path_id = id
        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame) or self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.pos, self.vel,
                                              self.epoch,
                                              self.origin.shape,
                                              self.frame.shape)

        # Update waypoint and path_id; register only if necessary
        self.register()

    ########################################

    def event_at_time(self, time, quick=None):
        """Return an Event corresponding to a specified time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity on the path.
        """

        return Event(time, (self.pos + (time-self.epoch) * self.vel, self.vel),
                           self.origin, self.frame)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_LinearPath(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
