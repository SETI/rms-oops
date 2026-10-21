##########################################################################################
# tests/host/test_host_overrides.py: The tracker option of Host._apply_overrides
##########################################################################################

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from oops         import Body, Frame, Path
from oops.cadence import SnapCadence
from oops.frame   import TrackerFrame
from oops.host    import Host

DAY = 86400.
TSTART = 0.
TEXP = 100.


@pytest.fixture(scope='module', autouse=True)
def _solar_system() -> Iterator[None]:
    Body.reset_registry()
    Body.define_solar_system('1990-01-01', '2020-01-01')
    yield
    Body.reset_registry()


def _params(target: Any) -> dict[str, Any]:
    """The constructor inputs of an observation from Earth, before any overrides.

    Parameters:
        target: The target of the observation.

    Returns:
        dict[str, Any]: The inputs.
    """

    return {'target': target, 'path': 'EARTH', 'frame': 'IAU_MARS',
            'cadence': SnapCadence(TSTART, TEXP)}


def _tracker_frame(target: Any, tracker: str = 'start') -> TrackerFrame:
    """The frame that the tracker option assigns to an observation of `target`.

    Parameters:
        target: The target of the observation.
        tracker: The tracker option.

    Returns:
        TrackerFrame: The frame of the observation after the overrides.
    """

    params = Host._apply_overrides(_params(target), 'dir/file.img', tracker=tracker)
    frame = params['frame']
    assert isinstance(frame, TrackerFrame)
    return frame


@pytest.mark.parametrize('target', ['NONE', 'none', 'None'])
def test_a_target_of_none_gives_a_tracker_without_a_target(target: str) -> None:
    """A target of "NONE", in any case, gives a TrackerFrame with no target path."""

    frame = _tracker_frame(target)

    assert frame._target_path is None


def test_a_target_of_none_gives_inertial_pointing() -> None:
    """The orientation of the frame at the epoch holds at every later time."""

    frame = _tracker_frame('NONE')
    expected = Frame.as_frame('IAU_MARS').wrt('J2000').transform_at_time(TSTART).matrix

    for t in (TSTART, TSTART + DAY, TSTART + 10. * DAY):
        matrix = frame.transform_at_time(t).matrix
        assert np.abs(matrix.vals - expected.vals).max() < 1.e-15


@pytest.mark.parametrize('target', ['MARS', 'mars'], ids=['upper', 'lower'])
def test_a_target_name_tracks_the_path_of_that_body(target: str) -> None:
    """A body name is converted to the path of that body."""

    frame = _tracker_frame(target)

    assert frame._target_path == Path.as_waypoint('MARS')


def test_a_target_body_tracks_its_path() -> None:
    """A Body is converted to its path."""

    frame = _tracker_frame(Body.lookup('MARS'))

    assert frame._target_path == Path.as_waypoint('MARS')


def test_a_tracked_target_holds_a_fixed_direction() -> None:
    """A tracked target is not inertial: the frame turns to follow it."""

    frame = _tracker_frame('MARS')

    start = frame.transform_at_time(TSTART).matrix.vals
    later = frame.transform_at_time(TSTART + 10. * DAY).matrix.vals
    assert np.abs(later - start).max() > 1.e-6


def test_the_observer_is_the_path_of_the_observation() -> None:
    """The TrackerFrame observes from the path of the observation."""

    frame = _tracker_frame('MARS')

    assert frame._observer_path == Path.as_waypoint('EARTH')


@pytest.mark.parametrize(('tracker', 'epoch'),
                         [('start', TSTART), ('midtime', TSTART + TEXP / 2.),
                          ('end', TSTART + TEXP)])
def test_the_tracker_option_sets_the_epoch(tracker: str, epoch: float) -> None:
    """The epoch is the start, middle or end of the exposure."""

    frame = _tracker_frame('NONE', tracker)

    assert frame._epoch == epoch


def test_an_unknown_tracker_option_is_rejected() -> None:
    """A tracker option other than start, midtime and end is an error."""

    with pytest.raises(ValueError, match='invalid tracker'):
        Host._apply_overrides(_params('NONE'), 'dir/file.img', tracker='finish')


def test_an_unknown_target_name_is_rejected() -> None:
    """A target that names no body cannot be tracked."""

    with pytest.raises(KeyError, match='NOT_A_BODY'):
        Host._apply_overrides(_params('NOT_A_BODY'), 'dir/file.img', tracker='start')


def test_a_target_override_is_tracked() -> None:
    """The target given as an override, not the default target, is the one tracked."""

    params = Host._apply_overrides(_params('MARS'), 'dir/file.img', target='NONE',
                                   tracker='start')

    assert params['target'] == 'NONE'
    assert params['frame']._target_path is None


def test_without_the_tracker_option_the_frame_is_unchanged() -> None:
    """No TrackerFrame is assigned unless the tracker option is given."""

    params = Host._apply_overrides(_params('NONE'), 'dir/file.img')

    assert params['frame'] == 'IAU_MARS'

##########################################################################################
