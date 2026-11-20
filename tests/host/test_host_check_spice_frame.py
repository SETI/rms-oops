##########################################################################################
# tests/host/test_host_check_spice_frame.py: Host._check_spice_frame
##########################################################################################

from typing import Any

import pytest

from oops       import Frame, OopsSpiceError
from oops.frame import SpiceErrorFrame
from oops.host  import Host

SPICE_TEXT = 'SPICE(CKINSUFFDATA) -- Insufficient data for frame -82360'


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


def _check(frame: Any, **kwargs: Any) -> Any:
    return Host._check_spice_frame(frame, 0., description='no NAC pointing',
                                   filepath='image.img', **kwargs)


def test_a_frame_that_spice_supplies_is_returned_unchanged() -> None:
    assert _check('J2000', defer=True) == 'J2000'


def test_missing_pointing_raises_the_spice_error() -> None:
    with pytest.raises(OopsSpiceError, match=r'for image\.img: SPICE\(CKINSUFFDATA\)$'):
        _check(_FailingFrame(RuntimeError(SPICE_TEXT)))


def test_deferred_missing_pointing_gives_a_spice_error_frame() -> None:
    frame = _check(_FailingFrame(RuntimeError(SPICE_TEXT)), defer=True)

    assert isinstance(frame, SpiceErrorFrame)


def test_an_unregistered_frame_id_is_not_a_spice_error() -> None:
    """The KeyError of an unknown frame ID, which cspyce would also raise for an unknown
    name, is the caller's error and propagates unchanged."""

    with pytest.raises(KeyError, match='NOT_A_FRAME'):
        _check('NOT_A_FRAME')


def test_an_error_that_cspyce_does_not_raise_propagates() -> None:
    with pytest.raises(TypeError, match='not about SPICE'):
        _check(_FailingFrame(TypeError('not about SPICE')))

##########################################################################################
