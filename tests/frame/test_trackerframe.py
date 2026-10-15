##########################################################################################
# tests/frame/test_trackerframe.py: Subclass TrackerFrame of class Frame
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Vector3
from oops       import Body, Event, Frame, Path
from oops.frame import TrackerFrame


@pytest.fixture(autouse=True)
def _solar_system():
    Body.reset_registry()
    Body.define_solar_system('1990-01-01', '2020-01-01')

def test_trackerframe():
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
        assert abs(obs_event.arr_ap.unit() - start_arr) < 1.e-6

    # Try the test all at once
    t = np.arange(0,30*DAY,DAY/40)
    obs_event = Event(t, Vector3.ZERO, "EARTH", "TEST")
    (path_event, obs_event) = mars.photon_to_event(obs_event)
    assert abs(obs_event.arr_ap.unit() - start_arr).max() < 1.e-6


DAY = 86400.


def test_without_a_target_the_pointing_is_inertial() -> None:
    """A TrackerFrame with no target keeps the orientation its frame has at epoch."""

    frame = TrackerFrame('IAU_MARS', None, 'EARTH', 0.)
    expected = Frame.as_frame('IAU_MARS').wrt('J2000').transform_at_time(0.).matrix

    for t in (0., 0.3 * DAY, 10. * DAY):
        matrix = frame.transform_at_time(t).matrix
        assert np.abs(matrix.vals - expected.vals).max() < 1.e-15


def test_the_frame_of_inertial_pointing_does_not_rotate() -> None:
    """The wrapped frame rotates, but the inertial TrackerFrame does not."""

    frame = TrackerFrame('IAU_MARS', None, 'EARTH', 0.)
    mars = Frame.as_frame('IAU_MARS').wrt('J2000')
    times = np.arange(0., DAY, DAY / 8.)

    start = mars.transform_at_time(0.).matrix.vals
    assert np.abs(mars.transform_at_time(times[3]).matrix.vals - start).max() > 0.1
    assert np.all(frame.transform_at_time(times).matrix.vals
                  == frame.transform_at_time(0.).matrix.vals)


def test_inertial_pointing_has_no_target_path() -> None:
    """The target of an inertial TrackerFrame is shown as None."""

    frame = TrackerFrame('J2000', None, 'EARTH', 0.)

    assert 'target = None' in frame.show(2)


def test_inertial_pointing_survives_pickling() -> None:
    """A pickled inertial TrackerFrame restores the same orientation."""

    frame = TrackerFrame('IAU_MARS', None, 'EARTH', 0.)
    copy = pickle.loads(pickle.dumps(frame))

    assert copy.transform_at_time(DAY).matrix == frame.transform_at_time(DAY).matrix
##########################################################################################
