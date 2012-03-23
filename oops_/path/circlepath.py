################################################################################
# oops_/path/orbit.py: Subclass CirclePath of class Path
################################################################################

import numpy as np
import cspice

from oops_.path.path_ import Path, Waypoint, RotatedPath
from oops_.array.all import *
from oops_.config import QUICK
from oops_.event import Event
import oops_.registry as registry
import oops_.spice_support as spice

class CirclePath(Path):
    """Subclass CirclePath of class Path that moves in uniform circular motion
    about another path, in an orientation defined by one of the z-axis of a
    given frame.."""

    def __init__(self, radius, lon, rate, epoch, origin, frame, id=None):
        """Constructor for an CirclPath.

        Input:
            radius      radius of the path, km.
            lon         longitude of the path at epoch, measured from the
                        x-axis of the frame, toward the y-axis, in radians.
            rate        rate of circular motion, radians/second.

            epoch       the time TDB relative to which all orbital elements are
                        defined.
            origin      the path or ID of the center of the circle.
            frame       the frame or ID of the frame in which the circular
                        motion is defined.
            id          the name under which to register the new path; None to
                        use a temporary path ID.

        Note: The shape of the Path object returned is defined by broadcasting
        together the shapes of all the orbital elements plus the epoch.
        """

        if id is None:
            self.path_id = registry.temporary_path_id()
        else:
            self.path_id = id

        self.origin_id = registry.as_path_id(origin)
        self.frame_id = registry.as_frame_id(frame)

        self.epoch = Scalar.as_scalar(epoch)
        self.shape = Array.broadcast_shape((radius, lon, rate, epoch))

        # Interpret the elements
        self.radius = Scalar.as_standard(radius)
        self.lon    = Scalar.as_standard(lon)
        self.rate   = Scalar.as_standard(rate)

        self.register()

########################################

    def event_at_time(self, time, quick=QUICK):
        """Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        lon = self.lon + self.rate * (Scalar.as_scalar(time) - self.epoch)
        r_cos_lon = self.radius * lon.cos()
        r_sin_lon = self.radius * lon.sin()

        pos = Vector3.from_scalars(r_cos_lon, r_sin_lon, 0.)
        vel = Vector3.from_scalars(-r_sin_lon * self.rate,
                                    r_cos_lon * self.rate, 0.)

        return Event(time, pos, vel, self.origin_id, self.frame_id)

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
