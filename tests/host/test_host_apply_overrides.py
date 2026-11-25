##########################################################################################
# tests/host/test_host_apply_overrides.py: The standard overrides of Host._apply_overrides
##########################################################################################

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oops         import Matrix3
from oops.cadence import SnapCadence, TimeShift
from oops.frame   import Cmatrix, FrameShift, Navigation
from oops.host    import Host

TEXP = 100.
TURN = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def _params() -> dict[str, Any]:
    """The constructor inputs of an observation from J2000, before any overrides."""

    return {'target': 'NONE', 'path': 'SSB', 'frame': 'J2000',
            'cadence': SnapCadence(0., TEXP)}


def test_a_standard_override_replaces_the_default() -> None:
    params = Host._apply_overrides(_params(), frame='SOME_FRAME')

    assert params['frame'] == 'SOME_FRAME'


def test_a_timeshift_couples_later_letters_to_the_first() -> None:
    params = Host._apply_overrides(_params(), timeshift=(5., 'tf'))

    assert isinstance(params['cadence'], TimeShift)
    assert params['cadence'].dt == 5.
    assert isinstance(params['frame'], FrameShift)
    assert params['frame'].dt == 5.
    assert params['frame'].link is params['cadence']


@pytest.mark.parametrize(('letters', 'message'),
                         [('', 'must specify one or more of T/P/F'),
                          ('TX', 'invalid timeshift letters: TX')],
                         ids=['empty', 'unknown'])
def test_invalid_timeshift_letters_are_rejected(letters: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Host._apply_overrides(_params(), timeshift=(5., letters))


def test_parallel_and_tracker_cannot_both_be_given() -> None:
    parallel =SimpleNamespace(frame='J2000')

    with pytest.raises(ValueError, match='parallel and tracker cannot both be'):
        Host._apply_overrides(_params(), parallel=parallel, tracker='start')


def test_a_parallel_observation_fixes_the_frame_relative_to_its_frame() -> None:
    params = _params()
    params['frame'] = Cmatrix(Matrix3(TURN), 'J2000')
    params = Host._apply_overrides(params, parallel=SimpleNamespace(frame='J2000'))
    frame = params['frame']

    assert isinstance(frame, Cmatrix)
    assert np.abs(frame.transform_at_time(0.).matrix.vals - TURN).max() < 1.e-15


def test_navigation_wraps_the_frame() -> None:
    params = Host._apply_overrides(_params(), navigation=(0., 0.))

    assert isinstance(params['frame'], Navigation)

##########################################################################################
