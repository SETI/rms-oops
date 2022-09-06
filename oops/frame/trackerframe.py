################################################################################
# oops/frame/trackerframe.py: Subclass TrackerFrame of class Frame
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from .           import Frame
from ..path      import Path, AliasPath
from ..transform import Transform
from ..event     import Event

class TrackerFrame(Frame):
    """A Frame subclass that ensures, via a small rotation, that a designated
    target path will remain in a fixed direction.

    The primary use of this frame is for observing moving targets with HST.
    Normally, HST images of the same target, obtained during the same visit and
    orbit, will have a common pointing offset and can be navigated as a group.
    This is not generally true when using the pointing information in the FITS
    headers, because that pointing refers to the start time of each frame rather
    than the midtime.
    """

    PACKRAT_ARGS = ['fixed_frame', 'target_path', 'observer_path', 'frame_id']

    #===========================================================================
    def __init__(self, frame, target, observer, epoch, id=None):
        """Constructor for a Tracker Frame.

        Input:
            frame       the frame that will be modified to enable tracking, or
                        its frame ID. Must be inertial.
            target      the target's path or path ID.
            observer    the observer's path or path ID.
            epoch       the epoch for which the given frame is defined.
            id          the ID to use; None to use a temporary ID.
        """

        self.fixed_frame = Frame.as_frame(frame)
        self.target_path = Path.as_path(target)
        self.observer_path = Path.as_path(observer)
        self.epoch = epoch

        assert self.fixed_frame.shape == ()
        assert self.target_path.shape == ()
        assert self.observer_path.shape == ()

        # Required attributes
        self.frame_id  = id
        self.reference = self.fixed_frame.reference
        self.origin    = self.fixed_frame.origin
        self.shape     = ()
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register()

        # Determine the apparent direction to the target path at epoch
        obs_event = Event(epoch, Vector3.ZERO, self.observer_path, Frame.J2000)
        (path_event, obs_event) = self.target_path.photon_to_event(obs_event)
        self.trackpoint = obs_event.neg_arr_ap.unit()

        # Determine the transform at epoch
        fixed_xform = self.fixed_frame.transform_at_time(self.epoch)
        self.reference_xform = Transform(fixed_xform.matrix, Vector3.ZERO,
                                         self.wayframe, self.reference,
                                         self.origin)
        assert fixed_xform.omega == Vector3.ZERO

        # Convert the matrix to three axis vectors
        self.reference_rows = Vector3(self.reference_xform.matrix.values)

        # Prepare to cache the most recently used transform
        self.cached_time = None
        self.cached_xform = None
        dummy = self.transform_at_time(self.epoch)  # cache initialized

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """The Transform into the this Frame at a Scalar of times."""

        if time == self.cached_time:
            return self.cached_xform

        # Determine the needed rotation
        obs_event = Event(time, Vector3.ZERO, self.observer_path, Frame.J2000)
        (path_event, obs_event) = self.target_path.photon_to_event(obs_event)
        newpoint = obs_event.neg_arr_ap.unit()

        rotation = self.trackpoint.cross(newpoint)
        rotation = rotation.reshape(rotation.shape + (1,))

        # Rotate the three axis vectors accordingly
        new_rows = self.reference_rows.spin(rotation)
        xform = Transform(Matrix3(new_rows.vals),
                          Vector3.ZERO,     # neglect the slow frame rotation
                          self.wayframe, self.reference, self.origin)

        # Cache the most recently used transform
        self.cached_time = time
        self.cached_xform = xform
        return xform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_TrackerFrame(unittest.TestCase):

    def setUp(self):
        from ..body import Body
        import oops.spice_support as spice

        Body.reset_registry()
        Body.define_solar_system('1990-01-01', '2020-01-01')

    def tearDown(self):
        pass

    def runTest(self):

        tracker = TrackerFrame("J2000", "MARS", "EARTH", 0., id="TEST")
        mars = AliasPath("MARS")

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
