##########################################################################################
# tests/path/test_linearpath.py
##########################################################################################

import pickle

import pytest

from polymath import Scalar, Vector3
from oops.frame import Frame
from oops.path  import LinearPath, Path

POS   = (1.e5, 2.e5, -3.e5)
VEL   = (1., -2., 3.)
EPOCH = 1000.


def linear_path() -> LinearPath:
    """A LinearPath through POS at EPOCH, moving at VEL."""

    return LinearPath((POS, VEL), EPOCH, Path.SSB, frame=Frame.J2000)


def test_linearpath_position_is_linear_in_time() -> None:
    """The position is the epoch position plus the velocity times the elapsed time."""

    path = linear_path()

    assert path.event_at_time(Scalar(EPOCH)).pos == Vector3(POS)

    for dt in (0., 10., -10., 3600.):
        expected = Vector3(POS) + Vector3(VEL) * dt
        assert path.event_at_time(Scalar(EPOCH + dt)).pos == expected


def test_linearpath_velocity_is_constant() -> None:
    """The motion is linear, so the velocity does not depend on time."""

    path = linear_path()

    for time in (EPOCH, EPOCH + 500., EPOCH - 500.):
        assert path.event_at_time(Scalar(time)).vel == Vector3(VEL)


def test_linearpath_velocity_may_be_given_as_a_derivative() -> None:
    """A Vector3 carrying a 'd_dt' derivative describes the same path as a (pos, vel)
    tuple."""

    pos = Vector3(POS)
    pos.insert_deriv('t', Vector3(VEL))
    by_deriv = LinearPath(pos, EPOCH, Path.SSB, frame=Frame.J2000)
    by_tuple = linear_path()

    time = Scalar(EPOCH + 250.)
    assert by_deriv.event_at_time(time).pos == by_tuple.event_at_time(time).pos
    assert by_deriv.event_at_time(time).vel == by_tuple.event_at_time(time).vel


def test_linearpath_with_zero_velocity_is_stationary() -> None:
    """A zero velocity leaves the position unchanged at every time."""

    path = LinearPath((POS, (0., 0., 0.)), EPOCH, Path.SSB, frame=Frame.J2000)

    assert path.event_at_time(Scalar(EPOCH - 1.e6)).pos == Vector3(POS)
    assert path.event_at_time(Scalar(EPOCH + 1.e6)).pos == Vector3(POS)


def test_linearpath_frame_defaults_to_the_origin_frame() -> None:
    """With no frame given, coordinates are expressed in the origin path's frame."""

    implied = LinearPath((POS, VEL), EPOCH, Path.SSB)

    assert implied.frame == Path.SSB.frame


def test_linearpath_is_shaped_by_its_arguments() -> None:
    """Multidimensional positions give the path that same shape."""

    positions = Vector3([(1.e5, 0., 0.), (0., 2.e5, 0.)])
    velocities = Vector3([(1., 0., 0.), (0., 1., 0.)])
    path = LinearPath((positions, velocities), EPOCH, Path.SSB, frame=Frame.J2000)

    assert path.shape == (2,)

    event = path.event_at_time(Scalar(EPOCH + 100.))
    assert event.pos == positions + velocities * 100.


def test_linearpath_epoch_may_be_shaped() -> None:
    """A per-element epoch shifts each element's reference time independently."""

    positions = Vector3([(0., 0., 0.), (0., 0., 0.)])
    velocities = Vector3([(1., 0., 0.), (1., 0., 0.)])
    path = LinearPath((positions, velocities), Scalar([0., 100.]), Path.SSB,
                      frame=Frame.J2000)

    assert path.event_at_time(Scalar(100.)).pos == Vector3([(100., 0., 0.),
                                                            (0., 0., 0.)])


def test_linearpath_rejects_unregistered_ids() -> None:
    """An ID string that names no registered Path or Frame raises KeyError."""

    with pytest.raises(KeyError):
        LinearPath((POS, VEL), EPOCH, 'NOT_A_REGISTERED_PATH')

    with pytest.raises(KeyError):
        LinearPath((POS, VEL), EPOCH, Path.SSB, frame='NOT_A_REGISTERED_FRAME')


def test_linearpath_registration() -> None:
    """A path_id registers the path under that name."""

    path = LinearPath((POS, VEL), EPOCH, Path.SSB, frame=Frame.J2000,
                      path_id='TEST_LINEAR_PATH')
    try:
        assert path.path_id == 'TEST_LINEAR_PATH'
        registered = Path.as_path('TEST_LINEAR_PATH')
        assert registered.path_id == 'TEST_LINEAR_PATH'
        assert registered.event_at_time(Scalar(EPOCH)).pos == Vector3(POS)
    finally:
        Path._reset_caches()


def test_linearpath_pickle() -> None:
    """Pickling restores the position, velocity, epoch, origin, and frame."""

    path = linear_path()
    restored = pickle.loads(pickle.dumps(path))
    time = Scalar(EPOCH + 42.)

    assert isinstance(restored, LinearPath)
    assert restored.origin == path.origin
    assert restored.frame == path.frame
    assert restored.event_at_time(time).pos == path.event_at_time(time).pos
    assert restored.event_at_time(time).vel == path.event_at_time(time).vel


def test_linearpath_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    path = linear_path()
    state = path.__getstate__()

    copied = Path.__new__(LinearPath)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
