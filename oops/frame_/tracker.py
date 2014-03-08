################################################################################
# oops/frame_/tracker.py: Subclass Tracker of class Frame
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.path_.path   import Path, QuickPath, Waypoint
from oops.transform    import Transform
from oops.event        import Event

import oops.registry as registry
import oops.config   as config

class Tracker(Frame):
    """Tracker is a Frame subclass that ensures, via a small rotation, that a
    designated target path will stay in a fixed direction over a reasonably
    short time interval. ("Reasonably short" is defined by the requirement that
    the light travel time to the target not change significantly.)
    """

    def __init__(self, frame, target, observer, epoch, id=None):
        """Constructor for a Tracker Frame.

        Input:
            frame       the frame that will be modified to enable tracking. Must
                        be inertial.
            target      the name of the target path.
            observer    the name of the observer path.
            epoch       the epoch for which the given frame is defined.
            id          the ID to use; None to use a temporary ID.
        """

        self.fixed_frame = registry.as_frame(frame)
        self.target_path = registry.as_path(target)
        self.observer_path = registry.as_path(observer)
        self.epoch = epoch

        assert self.target_path.shape == []
        assert self.observer_path.shape == []
        self.fixed_frame.shape == []

        self.shape = []

        obs_event = Event(epoch, Vector3.ZERO, Vector3.ZERO,
                          self.observer_path, "J2000")
        path_event = self.target_path.photon_to_event(obs_event)
        self.trackpoint = -obs_event.arr.unit()

        if id is None:
            self.frame_id = registry.temporary_frame_name()
        else:
            self.frame_id = id

        self.reference_id = self.fixed_frame.reference_id
        self.origin_id = self.fixed_frame.origin_id

        fixed_xform = self.fixed_frame.transform_at_time(self.epoch)
        self.reference_xform = Transform(fixed_xform.matrix, Vector3.ZERO,
                                         self.frame_id, self.reference_id,
                                         self.origin_id)
        assert fixed_xform.omega == Vector3.ZERO

        # Convert the matrix to three axis vectors
        self.reference_rows = Vector3(self.reference_xform.matrix.vals)

        # Prepare to cache the most recently used transform
        self.cached_time = None
        self.cached_xform = None

        self.reregister()

        # Fill cache
        dummy = self.transform_at_time(self.epoch)

    ########################################

    def transform_at_time(self, time, quick=None):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        if time == self.cached_time: return self.cached_xform

        # Determine the needed rotation
        obs_event = Event(time, Vector3.ZERO, Vector3.ZERO,
                          self.observer_path, "J2000")
        path_event = self.target_path.photon_to_event(obs_event)
        newpoint = -obs_event.arr.unit()

        rotation = self.trackpoint.cross(newpoint)
        rotation = rotation.reshape(rotation.shape + [1])

        # Rotate the three axis vectors accordingly
        new_rows = self.reference_rows.spin(rotation)
        xform = Transform(Matrix3(new_rows.vals),
                          Vector3.ZERO, # neglect the slow frame rotation
                          self.frame_id, self.reference_id, self.origin_id)

        # Cache the most recently used transform
        self.cached_time = time
        self.cached_xform = xform
        return xform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Tracker(unittest.TestCase):

    def runTest(self):

        import oops.body as body
        import oops.spice_support as spice

        registry.initialize()
        spice.initialize()

        body.define_solar_system("1999-01-01", "2002-01-01")

        tracker = Tracker("J2000", "MARS", "EARTH", 0., id="TEST")
        mars = Waypoint("MARS")

        obs_event = Event(0., Vector3.ZERO, Vector3.ZERO, "EARTH", "J2000")
        path_event = mars.photon_to_event(obs_event)
        start_arr = obs_event.arr.unit()

        # Track Mars for 30 days
        DAY = 86400
        for t in range(0,30*DAY,DAY):
            obs_event = Event(t, Vector3.ZERO, Vector3.ZERO, "EARTH", "TEST")
            path_event = mars.photon_to_event(obs_event)
            self.assertTrue(abs(obs_event.arr.unit() - start_arr) < 1.e-6)

        # Try the test all at once
        t = np.arange(0,30*DAY,DAY/40)
        obs_event = Event(t, Vector3.ZERO, Vector3.ZERO, "EARTH", "TEST")
        path_event = mars.photon_to_event(obs_event)
        self.assertTrue(abs(obs_event.arr.unit() - start_arr) < 1.e-6)

        registry.initialize()
        spice.initialize()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
