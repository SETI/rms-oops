################################################################################
# tests/frame/test_trackerframe.py: Subclass TrackerFrame of class Frame
################################################################################

import numpy as np
import unittest

from polymath   import Vector3
from oops       import Body, Event, Path
from oops.frame import TrackerFrame


class Test_TrackerFrame(unittest.TestCase):

    def setUp(self):
        Body.reset_registry()
        Body.define_solar_system('1990-01-01', '2020-01-01')

    def tearDown(self):
        pass

    def runTest(self):

        _ = TrackerFrame("J2000", "MARS", "EARTH", 0., frame_id="TEST")
        mars = Path.as_path("MARS")

        obs_event = Event(0., Vector3.ZERO, "EARTH", "J2000")
        (path_event, obs_event) = mars.photon_to_event(obs_event)
        start_arr = obs_event.arr_ap.unit()

        # Track Mars for 30 days
        DAY = 86400
        for t in range(0,30*DAY,DAY):
            obs_event = Event(t, Vector3.ZERO, "EARTH", "TEST")
            (path_event, obs_event) = mars.photon_to_event(obs_event)
            self.assertTrue(abs(obs_event.arr_ap.unit() - start_arr) < 1.e-6)

        # Try the test all at once
        t = np.arange(0,30*DAY,DAY/40)
        obs_event = Event(t, Vector3.ZERO, "EARTH", "TEST")
        (path_event, obs_event) = mars.photon_to_event(obs_event)
        self.assertTrue(abs(obs_event.arr_ap.unit() - start_arr).max() < 1.e-6)

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
