##########################################################################################
# tests/frame/test_twovectorframe.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame, TwoVectorFrame
from oops.path  import Path

HALFPI = np.pi / 2.


@pytest.fixture(autouse=True)
def _empty_registries():
    Frame._reset_caches()
    Path._reset_caches()
    yield
    Frame._reset_caches()
    Path._reset_caches()


def test_twovectorframe_aligned_with_the_reference() -> None:
    """Taking the reference frame's own axes reproduces the reference frame."""

    frame = TwoVectorFrame(Frame.J2000, Vector3.ZAXIS, 'z', Vector3.XAXIS, 'x')
    matrix = frame.transform_at_time(Scalar(0.)).matrix

    assert (matrix * Vector3.XAXIS).vals == pytest.approx([1., 0., 0.], abs=1.e-15)
    assert (matrix * Vector3.YAXIS).vals == pytest.approx([0., 1., 0.], abs=1.e-15)
    assert (matrix * Vector3.ZAXIS).vals == pytest.approx([0., 0., 1.], abs=1.e-15)


def test_twovectorframe_first_vector_defines_its_axis_exactly() -> None:
    """The first vector becomes the named axis of the new frame."""

    vector1 = Vector3((1., 1., 1.))
    frame = TwoVectorFrame(Frame.J2000, vector1, 'z', Vector3.XAXIS, 'x')
    matrix = frame.transform_at_time(Scalar(0.)).matrix

    # vector1, expressed in the new frame, points along its z-axis
    rotated = matrix * vector1.unit()
    assert rotated.vals == pytest.approx([0., 0., 1.], abs=1.e-15)


def test_twovectorframe_second_vector_sets_the_half_plane() -> None:
    """The second vector need not be perpendicular; only its half-plane matters."""

    vector1 = Vector3.ZAXIS
    perpendicular = TwoVectorFrame(Frame.J2000, vector1, 'z', Vector3((1., 0., 0.)), 'x')
    tilted = TwoVectorFrame(Frame.J2000, vector1, 'z', Vector3((3., 0., 5.)), 'x')

    assert tilted.transform_at_time(Scalar(0.)).matrix \
           == perpendicular.transform_at_time(Scalar(0.)).matrix

    # The length of the second vector is likewise irrelevant
    longer = TwoVectorFrame(Frame.J2000, vector1, 'z', Vector3((7., 0., 0.)), 'x')
    assert longer.transform_at_time(Scalar(0.)).matrix \
           == perpendicular.transform_at_time(Scalar(0.)).matrix


def test_twovectorframe_axes_may_be_named_or_numbered() -> None:
    """0/"x"/"X" all select the x-axis, and likewise for y and z."""

    reference = TwoVectorFrame(Frame.J2000, Vector3((1., 1., 0.)), 2,
                               Vector3.XAXIS, 0).transform_at_time(Scalar(0.)).matrix

    for axis1 in ('z', 'Z'):
        for axis2 in ('x', 'X'):
            frame = TwoVectorFrame(Frame.J2000, Vector3((1., 1., 0.)), axis1,
                                   Vector3.XAXIS, axis2)
            assert frame.transform_at_time(Scalar(0.)).matrix == reference


def test_twovectorframe_rejects_an_unknown_axis() -> None:
    """An unrecognized axis raises KeyError."""

    with pytest.raises(KeyError):
        TwoVectorFrame(Frame.J2000, Vector3.ZAXIS, 'w', Vector3.XAXIS, 'x')

    with pytest.raises(KeyError):
        TwoVectorFrame(Frame.J2000, Vector3.ZAXIS, 'z', Vector3.XAXIS, 'w')


def test_twovectorframe_rejects_an_unregistered_reference() -> None:
    """A reference ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        TwoVectorFrame('NOT_A_REGISTERED_FRAME', Vector3.ZAXIS, 'z', Vector3.XAXIS, 'x')


def test_twovectorframe_is_fixed_in_time() -> None:
    """A TwoVectorFrame is fixed, so its Transform does not depend on time."""

    frame = TwoVectorFrame(Frame.J2000, Vector3((0., 1., 1.)), 'z', Vector3.XAXIS, 'x')
    transform = frame.transform_at_time(Scalar(0.))

    assert transform.is_fixed
    assert transform.omega == Vector3.ZERO
    assert frame.transform_at_time(Scalar(1.e9)).matrix == transform.matrix

    # The Transform has the shape of the Frame, whatever the shape of the time
    assert frame.transform_at_time(Scalar([0., 10., 20.])).shape == frame.shape


def test_twovectorframe_may_be_multidimensional() -> None:
    """Arrays of vectors give the Frame that same shape."""

    vector1 = Vector3([(0., 0., 1.), (0., 1., 0.), (1., 0., 0.)])
    frame = TwoVectorFrame(Frame.J2000, vector1, 'z', Vector3.XAXIS, 'y')

    assert frame.shape == (3,)
    assert frame.transform_at_time(Scalar(0.)).shape == (3,)


def test_twovectorframe_rejects_unbroadcastable_vectors() -> None:
    """Vectors whose shapes cannot be broadcast together raise ValueError."""

    vector1 = Vector3([(0., 0., 1.), (0., 1., 0.), (1., 0., 0.)])
    vector2 = Vector3([(1., 0., 0.), (0., 1., 0.)])

    with pytest.raises(ValueError):
        TwoVectorFrame(Frame.J2000, vector1, 'z', vector2, 'x')


@pytest.mark.parametrize('pole, node', [((0., -1., 1.), 0.),
                                        ((1., 0., 1.), HALFPI),
                                        ((0., 1., 1.), np.pi),
                                        ((-1., 0., 1.), 3. * HALFPI)],
                         ids=['tilt_-y', 'tilt_+x', 'tilt_+y', 'tilt_-x'])
def test_twovectorframe_node_follows_the_pole(pole: tuple[float, float, float],
                                              node: float) -> None:
    """The ascending node lies a quarter turn ahead of the direction of the tilt."""

    frame = TwoVectorFrame(Frame.J2000, Vector3(pole), 'z', Vector3.XAXIS, 'x')

    assert frame.node_at_time(Scalar(0.)).vals == pytest.approx(node, abs=1.e-12)


def test_twovectorframe_node_is_fixed_in_time() -> None:
    """A TwoVectorFrame is fixed, so its node does not depend on time."""

    frame = TwoVectorFrame(Frame.J2000, Vector3((0., -1., 1.)), 'z', Vector3.XAXIS, 'x')
    node = frame.node_at_time(Scalar(0.))

    assert 0. <= node.vals < 2. * np.pi
    assert frame.node_at_time(Scalar(1.e9)) == node
    assert frame.node_at_time(Scalar([0., 10.])).shape == frame.shape


def test_twovectorframe_registration() -> None:
    """A frame_id registers the Frame under that name."""

    frame = TwoVectorFrame(Frame.J2000, Vector3.ZAXIS, 'z', Vector3.XAXIS, 'x',
                           frame_id='TEST_TWOVECTOR')

    assert frame.frame_id == 'TEST_TWOVECTOR'
    assert Frame.as_frame('TEST_TWOVECTOR').frame_id == 'TEST_TWOVECTOR'


def test_twovectorframe_pickle() -> None:
    """Pickling restores the two vectors, their axes, and the reference frame."""

    frame = TwoVectorFrame(Frame.J2000, Vector3((1., 2., 3.)), 'z',
                           Vector3((0., 1., 0.)), 'y')
    restored = pickle.loads(pickle.dumps(frame))

    assert isinstance(restored, TwoVectorFrame)
    assert restored.reference == frame.reference
    assert restored.transform_at_time(Scalar(0.)).matrix \
           == frame.transform_at_time(Scalar(0.)).matrix


def test_twovectorframe_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    frame = TwoVectorFrame(Frame.J2000, Vector3((1., 2., 3.)), 'z',
                           Vector3((0., 1., 0.)), 'y')
    state = frame.__getstate__()

    copied = Frame.__new__(TwoVectorFrame)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
