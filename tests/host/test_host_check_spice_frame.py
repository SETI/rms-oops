##########################################################################################
# tests/host/test_host_check_spice_frame.py: Host._check_spice_frame
##########################################################################################

import math
import re
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest

from polymath       import Matrix3, Scalar, Vector3
from oops           import Frame, OopsSpiceError, OopsValueError
from oops.cadence   import SnapCadence
from oops.frame     import Cmatrix
from oops.host      import Host
from oops.transform import Transform

SPICE_TEXT = 'SPICE(CKINSUFFDATA) -- Insufficient data for frame -82360'
CADENCE = SnapCadence(100., 10.)        # midtime 105
FLIP = np.diag([1., -1., -1.])


class _FailingFrame(Frame):
    """A frame whose evaluation raises a given exception, as a SpiceFrame raises a cspyce
    error when the C kernels have no record."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self._reference = Frame.J2000
        self._origin = None
        self._shape = ()
        self._register()

    def transform_at_time(self, time: Any, *, quick: Any = None) -> Any:
        raise self._error


class _CKFrame(Frame):
    """A frame relative to J2000 that the C kernels supply from `start` up to `end`, as a
    SpiceFrame does: at any other time it raises the error that cspyce raises when no
    C-kernel record covers the time."""

    def __init__(self, start: float = -math.inf, end: float = math.inf) -> None:
        self._start = start
        self._end = end
        self._reference = Frame.J2000
        self._origin = None
        self._shape = ()
        self._register()

    def transform_at_time(self, time: Any, *, quick: Any = None) -> Any:
        vals = Scalar.as_scalar(time).vals
        if np.any(vals < self._start) or np.any(vals >= self._end):
            raise RuntimeError(SPICE_TEXT)
        return Transform(Matrix3.IDENTITY, Vector3.ZERO, self, Frame.J2000)


def _camera(start: float = -math.inf, end: float = math.inf) -> Cmatrix:
    """A camera frame that is a fixed rotation over a frame from the C kernels, as the
    New Horizons LORRI and Cassini ISS frames are."""

    return Cmatrix(FLIP, reference=_CKFrame(start, end))


def _check(frame: Any, overrides: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    return Host._check_spice_frame(frame, CADENCE, overrides or {},
                                   description='no NAC pointing', filepath='image.img',
                                   **kwargs)


def test_a_frame_that_spice_supplies_passes() -> None:
    assert _check('J2000') is None


def test_missing_pointing_raises_the_spice_error() -> None:
    with pytest.raises(OopsSpiceError, match=r'for image\.img: SPICE\(CKINSUFFDATA\)$'):
        _check(_FailingFrame(RuntimeError(SPICE_TEXT)))


def test_an_unregistered_frame_id_is_not_a_spice_error() -> None:
    """The KeyError of an unknown frame ID, which cspyce would also raise for an unknown
    name, is the caller's error and propagates unchanged."""

    with pytest.raises(KeyError, match='NOT_A_FRAME'):
        _check('NOT_A_FRAME')


def test_an_error_that_cspyce_does_not_raise_propagates() -> None:
    with pytest.raises(TypeError, match='not about SPICE'):
        _check(_FailingFrame(TypeError('not about SPICE')))


def test_an_error_without_a_spice_code_propagates() -> None:
    """A ValueError is of a class that cspyce raises, but without a SPICE code in its text
    it is not about SPICE data."""

    with pytest.raises(ValueError, match='not about SPICE'):
        _check(_FailingFrame(ValueError('not about SPICE')))


def test_a_frame_whose_reference_spice_cannot_supply_raises() -> None:
    """A Cmatrix transforms from its reference without evaluating it, so the check must
    evaluate the frame relative to J2000."""

    frame = Cmatrix(np.eye(3), reference=_FailingFrame(RuntimeError(SPICE_TEXT)))

    with pytest.raises(OopsSpiceError, match=r'SPICE\(CKINSUFFDATA\)$'):
        _check(frame)

##########################################################################################
# The overrides: the check evaluates what the observation will evaluate
##########################################################################################

class _Case(NamedTuple):
    """One override, the C-kernel coverage of the frame, and whether the pointing fails.

    `camera` is True for a fixed rotation over the frame from the C kernels, False for
    the frame from the C kernels itself. The observation's cadence runs from 100 to 110
    seconds, so its midtime is 105, and every time shift is 3 seconds.
    """

    camera: bool
    overrides: dict[str, Any]
    start: float
    end: float
    fails: bool


INF = math.inf

# Stands for a parallel observation in CASES; each test builds its own, because the tests
# of other modules reset the Frame caches, and a Frame built when this module is imported
# would lose its cache entries
PARALLEL = 'parallel observation'


def _parallel() -> SimpleNamespace:
    """A parallel observation whose frame the C kernels always supply."""

    return SimpleNamespace(frame=Cmatrix(np.eye(3), reference=_CKFrame()))


def _overrides(case: _Case) -> dict[str, Any]:
    """The overrides of a case, with a parallel observation built for PARALLEL."""

    return {key: _parallel() if value is PARALLEL else value
            for (key, value) in case.overrides.items()}

CASES = {
    'midtime covered': _Case(True, {}, -INF, 106., False),
    'midtime not covered': _Case(True, {}, -INF, 105., True),
    'cadence override': _Case(True, {'cadence': SnapCadence(200., 4.)}, -INF, 110., True),
    'T moves the time': _Case(True, {'timeshift': (3., 'T')}, -INF, 107., True),
    'F leaves a camera at the midtime': _Case(True, {'timeshift': (3., 'F')},
                                              -INF, 107., False),
    'F gives a camera no coverage': _Case(True, {'timeshift': (3., 'F')},
                                          106., INF, True),
    'F moves a SPICE frame': _Case(False, {'timeshift': (3., 'F')}, -INF, 107., True),
    'F moves a SPICE frame into coverage': _Case(False, {'timeshift': (3., 'f')},
                                                 106., INF, False),
    'TF moves a camera once': _Case(True, {'timeshift': (3., 'TF')}, -INF, 109., False),
    'TF moves a SPICE frame twice': _Case(False, {'timeshift': (3., 'FT')},
                                          -INF, 110., True),
    'P leaves the time': _Case(False, {'timeshift': (3., 'P')}, -INF, 106., False),
    'tracker start': _Case(True, {'tracker': 'start'}, -INF, 101., False),
    'tracker end': _Case(True, {'tracker': 'end'}, -INF, 109., True),
    'tracker with F on a camera': _Case(True, {'tracker': 'start',
                                               'timeshift': (3., 'F')},
                                        -INF, 101., False),
    'tracker with F on a SPICE frame': _Case(False, {'tracker': 'start',
                                                     'timeshift': (3., 'F')},
                                             -INF, 102., True),
    'navigation': _Case(True, {'navigation': (0., 0.)}, -INF, 105., True),
    'parallel': _Case(True, {'parallel': PARALLEL}, -INF, 105., True),
    'parallel with T': _Case(True, {'parallel': PARALLEL, 'timeshift': (3., 'T')},
                             -INF, 107., True),
    'parallel with F on a camera': _Case(True, {'parallel': PARALLEL,
                                                'timeshift': (3., 'F')},
                                         -INF, 107., False),
}


def _case_frame(case: _Case) -> Frame:
    if case.camera:
        return _camera(case.start, case.end)
    return _CKFrame(case.start, case.end)


def _check_fails(case: _Case) -> bool:
    try:
        _check(_case_frame(case), _overrides(case))
    except OopsSpiceError:
        return True
    return False


def _observation_fails(case: _Case) -> bool:
    """True if the observation, built with the overrides and without a check, fails to
    evaluate its frame relative to J2000 at its midtime."""

    params = {'cadence': CADENCE, 'frame': _case_frame(case), 'path': 'SSB',
              'target': 'NONE'}
    try:
        Host._apply_overrides(params, None, **_overrides(case))
        frame = Frame.as_frame(params['frame'])
        frame.wrt(Frame.J2000).transform_at_time(params['cadence'].midtime)
    except RuntimeError:
        return True
    return False


@pytest.mark.parametrize('name', CASES)
def test_the_check_fails_where_the_pointing_fails(name: str) -> None:
    assert _check_fails(CASES[name]) is CASES[name].fails


@pytest.mark.parametrize('name', CASES)
def test_the_observation_fails_where_the_pointing_fails(name: str) -> None:
    """The cases describe the observation that the overrides build, so the check above
    follows the observation."""

    assert _observation_fails(CASES[name]) is CASES[name].fails


def test_a_camera_with_its_parallel_is_checked_to_their_common_frame() -> None:
    """A parallel override evaluates the frame relative to the parallel frame, so a frame
    from the C kernels beneath both is never evaluated for this observation."""

    shared = _CKFrame(end=105.)
    camera = Cmatrix(FLIP, reference=shared)
    parallel = SimpleNamespace(frame=Cmatrix(np.eye(3), reference=shared))

    assert _check(camera, {'parallel': parallel}) is None


def test_a_parallel_frame_that_spice_cannot_supply_raises() -> None:
    camera = _camera()
    parallel = SimpleNamespace(frame=Cmatrix(np.eye(3), reference=_CKFrame(end=105.)))

    with pytest.raises(OopsSpiceError, match=r'SPICE\(CKINSUFFDATA\)$'):
        _check(camera, {'parallel': parallel})


def test_the_message_gives_the_time_of_the_observation() -> None:
    """The midtime after a cadence time shift; a frame time shift is not added."""

    expected = OopsSpiceError.from_spice_error(RuntimeError(SPICE_TEXT),
                                               description='no NAC pointing', time=108.,
                                               filepath='image.img')
    with pytest.raises(OopsSpiceError, match=f'^{re.escape(str(expected))}$'):
        _check(_CKFrame(end=107.), {'timeshift': (3., 'TF')})


def test_parallel_and_tracker_together_are_refused() -> None:
    with pytest.raises(OopsValueError, match='parallel and tracker cannot both be '
                                             'specified'):
        _check(_camera(), {'parallel': _parallel(), 'tracker': 'start'})


def test_unknown_timeshift_letters_are_refused() -> None:
    with pytest.raises(OopsValueError, match='invalid timeshift letters: TX'):
        _check(_camera(), {'timeshift': (3., 'tx')})


def test_repeated_timeshift_letters_are_refused() -> None:
    """The override would shift the times twice, coupled to one fittable time shift."""

    with pytest.raises(OopsValueError, match='repeated timeshift letters: TFT'):
        _check(_camera(), {'timeshift': (3., 'tft')})


def test_an_unknown_tracker_is_refused() -> None:
    with pytest.raises(OopsValueError, match="invalid tracker, should be \"start\", "
                                             "\"midtime\", or \"end\": 'middle'"):
        _check(_camera(), {'tracker': 'middle'})

##########################################################################################
