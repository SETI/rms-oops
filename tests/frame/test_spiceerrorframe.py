##########################################################################################
# tests/frame/test_spiceerrorframe.py: SpiceErrorFrame
##########################################################################################

import pickle

import pytest

from polymath   import Matrix3
from oops       import OopsSpiceError
from oops.frame import Cmatrix, SpiceErrorFrame

CAUSE = RuntimeError('SPICE(CKINSUFFDATA) -- Insufficient data for frame -82360')


def _frame() -> SpiceErrorFrame:
    return SpiceErrorFrame(CAUSE, description='no NAC pointing', filepath='image.img')


def test_evaluation_raises_the_spice_error() -> None:
    with pytest.raises(OopsSpiceError,
                       match=r'^no NAC pointing at 2000-01-01T11:58:55\.816 '
                             r'for image\.img: SPICE\(CKINSUFFDATA\)$'):
        _frame().transform_at_time(0.)


def test_the_error_is_raised_from_the_spice_exception() -> None:
    with pytest.raises(OopsSpiceError) as info:
        _frame().transform_at_time(0.)

    assert info.value.__cause__ is CAUSE


def test_evaluation_at_several_times_names_no_time() -> None:
    with pytest.raises(OopsSpiceError, match='^no NAC pointing for image.img:'):
        _frame().transform_at_time([0., 1.])


def test_a_frame_relative_to_another_raises_the_spice_error() -> None:
    """The error is raised however the frame is reached, as through wrt()."""

    reference = Cmatrix(Matrix3.IDENTITY)
    with pytest.raises(OopsSpiceError, match='no NAC pointing'):
        _frame().wrt(reference).transform_at_time(0.)


def test_the_frame_survives_pickling() -> None:
    frame = pickle.loads(pickle.dumps(_frame()))

    assert frame.description == 'no NAC pointing'


def test_the_expanded_description_names_what_is_missing() -> None:
    assert _frame().show(2) == "SpiceErrorFrame('no NAC pointing')"

##########################################################################################
