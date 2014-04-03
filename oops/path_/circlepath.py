################################################################################
# oops/path_/circlepath.py: Subclass CirclePath of class Path
################################################################################

import numpy as np
from polymath import *

from oops.event        import Event
from oops.path_.path   import Path
from oops.frame_.frame import Frame

class CirclePath(Path):
    """A path describing uniform circular motion about another path.

    The orientation of the circle is defined by the z-axis of the given
    frame.
    """

    def __init__(self, radius, lon, rate, epoch, origin, frame=None, id=None):
        """Constructor for a CirclePath.

        Input:
            radius      radius of the path, km.
            lon         longitude of the path at epoch, measured from the
                        x-axis of the frame, toward the y-axis, in radians.
            rate        rate of circular motion, radians/second.

            epoch       the time TDB relative to which all orbital elements are
                        defined.
            origin      the path or ID of the center of the circle.
            frame       the frame or ID of the frame in which the circular
                        motion is defined; None to use the default frame of the
                        origin path.
            id          the name under which to register the new path; None to
                        leave the path unregistered.

        Note: The shape of the Path object returned is defined by broadcasting
        together the shapes of all the orbital elements plus the epoch.
        """

        # Interpret the elements
        self.epoch  = Scalar.as_scalar(epoch)
        self.radius = Scalar.as_scalar(radius)
        self.lon    = Scalar.as_scalar(lon)
        self.rate   = Scalar.as_scalar(rate)

        # Required attributes
        self.path_id = id or Path.temporary_path_id()
        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame) or self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.radius, self.lon,
                                              self.rate, self.epoch,
                                              self.origin.shape,
                                              self.frame.shape)

        if id:
            self.register()
        else:
            self.waypoint = self

    ########################################

    def event_at_time(self, time, quick=False):
        """Return an Event corresponding to a specified time on this path.

        Input:
            time    a time Scalar at which to evaluate the path.

        Return:     an Event object containing (at least) the time, position
                    and velocity on the path.
        """

        lon = self.lon + self.rate * (Scalar.as_scalar(time) - self.epoch)
        r_cos_lon = self.radius * lon.cos()
        r_sin_lon = self.radius * lon.sin()

        pos = Vector3.from_scalars(r_cos_lon, r_sin_lon, 0.)
        vel = Vector3.from_scalars(-r_sin_lon * self.rate,
                                    r_cos_lon * self.rate, 0.)

        return Event(time, (pos,vel), self.origin, self.frame)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_CirclePath(unittest.TestCase):

    def runTest(self):

        # Note: Unit testing is performed in surface/orbitplane.py

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
