################################################################################
# oops/path/circlepath.py: Subclass CirclePath of class Path
################################################################################

import numpy as np
from polymath import Qube, Scalar, Vector3

from .       import Path
from ..event import Event
from ..frame import Frame

class CirclePath(Path):
    """A path describing uniform circular motion about another path.

    The orientation of the circle is defined by the z-axis of the given
    frame.
    """

    PATH_IDS = {}   # path_id to use if a path already exists upon un-pickling

    #===========================================================================
    def __init__(self, radius, lon, rate, epoch, origin, frame=None,
                       path_id=None, unpickled=False):
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
            path_id     the name under which to register the new path; None to
                        leave the path unregistered.
            unpickled   True if this path has been read from a pickle file.

        Note: The shape of the Path object returned is defined by broadcasting
        together the shapes of all the orbital elements plus the epoch.
        """

        # Interpret the elements
        self.epoch  = Scalar.as_scalar(epoch)
        self.radius = Scalar.as_scalar(radius)
        self.lon    = Scalar.as_scalar(lon)
        self.rate   = Scalar.as_scalar(rate)

        # Required attributes
        self.path_id = path_id
        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame) or self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.radius, self.lon,
                                              self.rate, self.epoch,
                                              self.origin.shape,
                                              self.frame.shape)

        # Update waypoint and path_id; register only if necessary
        self.register(unpickled=unpickled)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.path_id in Path.WAYPOINT_REGISTRY):
                key = (self.radius.vals, self.lon.vals, self.rate.vals,
                       self.epoch.vals, origin.path_id, frame.frame_id)
                CirclePath.PATH_IDS[key] = self.path_id

    def __getstate__(self):
        return (self.radius, self.lon, self.rate, self.epoch, self.origin,
                self.frame, self.shape)

    def __setstate__(self, state):
        # If this path matches a pre-existing path, re-use its ID
        (radius, lon, rate, epoch, origin, frame, shape) = state
        if shape == ():
            key = (radius.vals, lon.vals, rate.vals, epoch.vals, origin.path_id,
                   frame.frame_id)
            path_id = CirclePath.PATH_IDS.get(key, None)
        else:
            path_id = None

        self.__init__(radius, lon, rate, epoch, origin, frame, path_id=path_id,
                      unpickled=True)

    #===========================================================================
    def event_at_time(self, time, quick=False):
        """An Event corresponding to a specified time on this path.

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

        np.random.seed(2787)

        # Note: Unit testing is performed in surface/orbitplane.py

        ####################################
        # __getstate__/__setstate__

        radius = 100000.
        lon = 5 * np.random.randn()
        rate = 0.001 * np.random.randn()
        epoch = 10. * 365. * 86400. * np.random.randn()
        origin = Path.SSB
        frame = Frame.J2000
        path = CirclePath(radius, lon, rate, epoch, origin, frame)
        state = path.__getstate__()

        copied = Path.__new__(CirclePath)
        copied.__setstate__(state)
        self.assertEqual(copied.__getstate__(), state)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
