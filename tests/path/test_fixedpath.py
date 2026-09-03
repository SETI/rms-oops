##########################################################################################
# tests/path/test_fixedpath.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath import Scalar, Vector3
from oops.frame import Frame
from oops.path  import FixedPath, Path

POS = (1.e5, 2.e5, -3.e5)


def test_fixedpath_position_is_independent_of_time() -> None:
    """The coordinates are fixed, so the same position is returned at every time."""

    path = FixedPath(POS, Path.SSB, Frame.J2000)

    for time in (0., 1.e6, -1.e6):
        event = path.event_at_time(Scalar(time))
        assert event.pos == Vector3(POS)
        assert event.time == Scalar(time)


def test_fixedpath_velocity_is_zero() -> None:
    """A fixed position does not move relative to its origin and frame."""

    path = FixedPath(POS, Path.SSB, Frame.J2000)
    event = path.event_at_time(Scalar(500.))

    assert event.vel == Vector3((0., 0., 0.))


def test_fixedpath_pos_accepts_any_arraylike() -> None:
    """A tuple, a list, an array, and a Vector3 all describe the same path."""

    for arg in (POS, list(POS), np.array(POS), Vector3(POS)):
        path = FixedPath(arg, Path.SSB, Frame.J2000)
        assert path.event_at_time(Scalar(0.)).pos == Vector3(POS)


def test_fixedpath_frame_defaults_to_the_origin_frame() -> None:
    """With no frame given, the coordinates are those of the origin path's frame."""

    explicit = FixedPath(POS, Path.SSB, Path.SSB.frame)
    implied  = FixedPath(POS, Path.SSB)

    assert implied.frame == explicit.frame
    assert implied.event_at_time(Scalar(0.)).pos == explicit.event_at_time(Scalar(0.)).pos


def test_fixedpath_is_shaped_by_its_position() -> None:
    """A multidimensional position gives the path that same shape."""

    positions = Vector3([(1.e5, 0., 0.), (0., 2.e5, 0.), (0., 0., 3.e5)])
    path = FixedPath(positions, Path.SSB, Frame.J2000)

    assert path.shape == (3,)
    assert path.event_at_time(Scalar(0.)).pos == positions


def test_fixedpath_broadcasts_time_against_its_shape() -> None:
    """Times and the path's own shape are broadcast together."""

    positions = Vector3([(1.e5, 0., 0.), (0., 2.e5, 0.), (0., 0., 3.e5)])
    path = FixedPath(positions, Path.SSB, Frame.J2000)
    event = path.event_at_time(Scalar([[0.], [10.]]))

    assert event.shape == (2, 3)


def test_fixedpath_rejects_unbroadcastable_time() -> None:
    """Shapes that cannot be broadcast together raise ValueError."""

    positions = Vector3([(1.e5, 0., 0.), (0., 2.e5, 0.), (0., 0., 3.e5)])
    path = FixedPath(positions, Path.SSB, Frame.J2000)

    # The Event is built lazily, so the mismatch surfaces when its shape is needed
    with pytest.raises(ValueError, match='incompatible dimension'):
        path.event_at_time(Scalar([0., 10.])).shape


def test_fixedpath_rejects_unregistered_ids() -> None:
    """An ID string that names no registered Path or Frame raises KeyError."""

    with pytest.raises(KeyError):
        FixedPath(POS, 'NOT_A_REGISTERED_PATH')

    with pytest.raises(KeyError):
        FixedPath(POS, Path.SSB, 'NOT_A_REGISTERED_FRAME')


def test_fixedpath_registration() -> None:
    """A path_id registers the path under that name; None leaves it unregistered."""

    path = FixedPath(POS, Path.SSB, Frame.J2000, path_id='TEST_FIXED_PATH')
    try:
        assert path.path_id == 'TEST_FIXED_PATH'
        registered = Path.as_path('TEST_FIXED_PATH')
        assert registered.path_id == 'TEST_FIXED_PATH'
        assert registered.event_at_time(Scalar(0.)).pos == Vector3(POS)
    finally:
        Path._reset_caches()


def test_fixedpath_pickle() -> None:
    """Pickling restores the position, the origin, and the frame."""

    path = FixedPath(POS, Path.SSB, Frame.J2000)
    restored = pickle.loads(pickle.dumps(path))

    assert isinstance(restored, FixedPath)
    assert restored.origin == path.origin
    assert restored.frame == path.frame
    assert restored.event_at_time(Scalar(7.)).pos == path.event_at_time(Scalar(7.)).pos


def test_fixedpath_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    path = FixedPath(POS, Path.SSB, Frame.J2000)
    state = path.__getstate__()

    copied = Path.__new__(FixedPath)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
