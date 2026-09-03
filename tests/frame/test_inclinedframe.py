##########################################################################################
# tests/frame/test_inclinedframe.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame, InclinedFrame
from oops.path  import Path

INC   = 0.1
NODE  = 0.5
RATE  = 1.e-6
EPOCH = 1000.


@pytest.fixture(autouse=True)
def _empty_registries():
    Frame._reset_caches()
    Path._reset_caches()
    yield
    Frame._reset_caches()
    Path._reset_caches()


def test_inclinedframe_node_precesses_at_the_given_rate() -> None:
    """The node advances linearly from its value at epoch."""

    frame = InclinedFrame(INC, NODE, RATE, EPOCH)

    assert frame.node_at_time(Scalar(EPOCH)).vals == pytest.approx(NODE)

    for dt in (0., 1000., 1.e5):
        expected = (NODE + RATE * dt) % (2. * np.pi)
        assert frame.node_at_time(Scalar(EPOCH + dt)).vals == pytest.approx(expected)


def test_inclinedframe_node_is_wrapped_into_zero_to_twopi() -> None:
    """Values always fall between 0 and 2*pi."""

    frame = InclinedFrame(INC, NODE, RATE, EPOCH)
    dt = 2. * np.pi / RATE          # exactly one full precession cycle

    nodes = frame.node_at_time(Scalar([EPOCH, EPOCH + dt, EPOCH + 10. * dt,
                                       EPOCH - 10. * dt]))
    assert np.all(nodes.vals >= 0.)
    assert np.all(nodes.vals < 2. * np.pi)


def test_inclinedframe_zero_inclination_is_the_reference_frame() -> None:
    """With no inclination and no despin, the frame's pole is the reference pole."""

    frame = InclinedFrame(0., NODE, RATE, EPOCH)
    matrix = frame.transform_at_time(Scalar(EPOCH)).matrix

    assert (matrix * Vector3.ZAXIS).vals == pytest.approx([0., 0., 1.], abs=1.e-15)


def test_inclinedframe_pole_is_tilted_by_the_inclination() -> None:
    """The frame's pole is inclined from the reference pole by exactly `inc`."""

    frame = InclinedFrame(INC, NODE, 0., EPOCH)
    matrix = frame.transform_at_time(Scalar(EPOCH)).matrix

    # The z-axis of the new frame, expressed in reference coordinates, is the last
    # row of the rotation matrix; its angle from the reference pole is `inc`
    pole = matrix.unrotate(Vector3.ZAXIS)
    assert pole.sep(Vector3.ZAXIS).vals == pytest.approx(INC)


def test_inclinedframe_despin_changes_the_x_axis_only() -> None:
    """Despinning leaves the pole alone but reorients the x- and y-axes."""

    despun = InclinedFrame(INC, NODE, RATE, EPOCH, despin=True)
    spun   = InclinedFrame(INC, NODE, RATE, EPOCH, despin=False)
    time = Scalar(EPOCH + 5.e5)

    despun_pole = despun.transform_at_time(time).matrix.unrotate(Vector3.ZAXIS)
    spun_pole   = spun.transform_at_time(time).matrix.unrotate(Vector3.ZAXIS)
    assert despun_pole.vals == pytest.approx(spun_pole.vals, abs=1.e-12)

    despun_x = despun.transform_at_time(time).matrix.unrotate(Vector3.XAXIS)
    spun_x   = spun.transform_at_time(time).matrix.unrotate(Vector3.XAXIS)
    assert despun_x.sep(spun_x).vals > 1.e-6


def test_inclinedframe_reference_defaults_to_j2000() -> None:
    """With no reference given, the inclined plane is defined relative to J2000."""

    assert InclinedFrame(INC, NODE, RATE, EPOCH).reference == Frame.J2000


def test_inclinedframe_rejects_an_unregistered_reference() -> None:
    """A reference ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        InclinedFrame(INC, NODE, RATE, EPOCH, reference='NOT_A_REGISTERED_FRAME')


def test_inclinedframe_may_be_multidimensional() -> None:
    """Arrays of orbital elements give the Frame that same shape."""

    frame = InclinedFrame(Scalar([0.1, 0.2, 0.3]), Scalar([0.5, 0.6, 0.7]), RATE, EPOCH)

    assert frame.shape == (3,)
    assert frame.node_at_time(Scalar(EPOCH)).shape == (3,)
    assert frame.transform_at_time(Scalar(EPOCH)).shape == (3,)


def test_inclinedframe_rejects_unbroadcastable_elements() -> None:
    """Elements whose shapes cannot be broadcast together raise ValueError."""

    with pytest.raises(ValueError):
        InclinedFrame(Scalar([0.1, 0.2, 0.3]), Scalar([0.5, 0.6]), RATE, EPOCH)


def test_inclinedframe_rejects_unbroadcastable_time() -> None:
    """A time whose shape cannot be broadcast against the frame raises ValueError."""

    frame = InclinedFrame(Scalar([0.1, 0.2, 0.3]), NODE, RATE, EPOCH)

    with pytest.raises(ValueError):
        frame.transform_at_time(Scalar([0., 10.]))


def test_inclinedframe_auto_generated_frame_id() -> None:
    """A frame_id of "+" appends "_INCLINED" to the reference frame's ID."""

    frame = InclinedFrame(INC, NODE, RATE, EPOCH, reference=Frame.J2000, frame_id='+')

    assert frame.frame_id == 'J2000_INCLINED'


def test_inclinedframe_registration() -> None:
    """A frame_id registers the Frame under that name."""

    frame = InclinedFrame(INC, NODE, RATE, EPOCH, frame_id='TEST_INCLINED')

    assert frame.frame_id == 'TEST_INCLINED'
    assert Frame.as_frame('TEST_INCLINED').frame_id == 'TEST_INCLINED'


def test_inclinedframe_pickle() -> None:
    """Pickling restores the inclination, node, rate, and epoch."""

    frame = InclinedFrame(INC, NODE, RATE, EPOCH)
    restored = pickle.loads(pickle.dumps(frame))
    time = Scalar(EPOCH + 3000.)

    assert isinstance(restored, InclinedFrame)
    assert restored.reference == frame.reference
    assert restored.node_at_time(time) == frame.node_at_time(time)
    assert restored.transform_at_time(time).matrix == frame.transform_at_time(time).matrix


def test_inclinedframe_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    frame = InclinedFrame(INC, NODE, RATE, EPOCH)
    state = frame.__getstate__()

    copied = Frame.__new__(InclinedFrame)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
