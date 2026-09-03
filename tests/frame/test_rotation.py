##########################################################################################
# tests/frame/test_rotation.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame, Rotation
from oops.path  import Path


@pytest.fixture(autouse=True)
def _empty_registries():
    Frame._reset_caches()
    Path._reset_caches()
    yield
    Frame._reset_caches()
    Path._reset_caches()


def test_rotation_about_z() -> None:
    """A rotation by 90 degrees about Z carries the X-axis onto the -Y axis."""

    frame = Rotation(np.pi / 2., 'z', Frame.J2000)
    matrix = frame.transform_at_time(Scalar(0.)).matrix

    assert (matrix * Vector3.XAXIS).vals == pytest.approx([0., -1., 0.], abs=1.e-15)
    assert (matrix * Vector3.YAXIS).vals == pytest.approx([1., 0., 0.], abs=1.e-15)
    assert (matrix * Vector3.ZAXIS).vals == pytest.approx([0., 0., 1.], abs=1.e-15)


def test_rotation_about_x_and_y() -> None:
    """Each axis leaves its own direction fixed."""

    about_x = Rotation(0.3, 'x', Frame.J2000).transform_at_time(Scalar(0.)).matrix
    assert (about_x * Vector3.XAXIS).vals == pytest.approx([1., 0., 0.], abs=1.e-15)

    about_y = Rotation(0.3, 'y', Frame.J2000).transform_at_time(Scalar(0.)).matrix
    assert (about_y * Vector3.YAXIS).vals == pytest.approx([0., 1., 0.], abs=1.e-15)


def test_rotation_axis_may_be_named_or_numbered() -> None:
    """0/"x"/"X" all select the x-axis, and likewise for y and z."""

    for axis, alternatives in ((0, ('x', 'X')), (1, ('y', 'Y')), (2, ('z', 'Z'))):
        reference = Rotation(0.4, axis, Frame.J2000).transform_at_time(Scalar(0.)).matrix
        for alt in alternatives:
            matrix = Rotation(0.4, alt, Frame.J2000).transform_at_time(Scalar(0.)).matrix
            assert matrix == reference


def test_rotation_rejects_an_unknown_axis() -> None:
    """An unrecognized axis raises KeyError."""

    with pytest.raises(KeyError):
        Rotation(0.4, 'w', Frame.J2000)


def test_rotation_rejects_an_unregistered_reference() -> None:
    """A reference ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        Rotation(0.4, 'z', 'NOT_A_REGISTERED_FRAME')


def test_rotation_is_fixed_in_time() -> None:
    """A Rotation is a fixed Frame, so its Transform does not depend on time."""

    frame = Rotation(0.4, 'z', Frame.J2000)
    transform = frame.transform_at_time(Scalar(0.))

    assert transform.is_fixed
    assert transform.omega == Vector3.ZERO
    assert frame.transform_at_time(Scalar(1.e6)).matrix == transform.matrix

    # The Transform has the shape of the Frame, whatever the shape of the time
    assert frame.transform_at_time(Scalar([0., 10., 20.])).shape == frame.shape


def test_rotation_angle_is_a_scalar() -> None:
    """The rotation angle is exposed as a Scalar, whatever form it was given in."""

    for arg in (0.4, Scalar(0.4), np.array(0.4)):
        frame = Rotation(arg, 'z', Frame.J2000)
        assert isinstance(frame.angle, Scalar)
        assert frame.angle == Scalar(0.4)


def test_rotation_may_be_multidimensional() -> None:
    """An array of angles gives the Frame that same shape."""

    angles = Scalar([0.1, 0.2, 0.3])
    frame = Rotation(angles, 'z', Frame.J2000)

    assert frame.shape == (3,)
    assert frame.angle == angles
    assert frame.transform_at_time(Scalar(0.)).shape == (3,)


def test_rotation_is_fittable() -> None:
    """The rotation angle is the fittable parameter of a Rotation."""

    frame = Rotation(0.4, 'z', Frame.J2000)

    assert frame.params == (0.4,)
    assert frame.nparams == 1
    assert not frame.is_frozen

    frame.set_params((0.9,))
    assert frame.params == (0.9,)
    assert frame.angle == Scalar(0.9)


def test_rotation_freeze() -> None:
    """freeze=True returns an object that can no longer be fitted."""

    frozen = Rotation(0.4, 'z', Frame.J2000, freeze=True)

    assert frozen.is_frozen
    assert frozen.angle == Scalar(0.4)


def test_rotation_tracks_another_rotation() -> None:
    """Given another Rotation, this object's angle always matches that one's."""

    source = Rotation(0.4, 'z', Frame.J2000)
    linked = Rotation(source, 'z', Frame.J2000)

    assert linked.angle == Scalar(0.4)

    source.set_params((1.1,))
    assert linked.angle == Scalar(1.1)


def test_rotation_auto_generated_frame_id() -> None:
    """A frame_id of "+" appends "_ROTATED" to the reference frame's ID."""

    frame = Rotation(0.4, 'z', Frame.J2000, frame_id='+')

    assert frame.frame_id == 'J2000_ROTATED'


def test_rotation_registration() -> None:
    """A frame_id registers the Frame under that name; None leaves it unregistered."""

    frame = Rotation(0.4, 'z', Frame.J2000, frame_id='TEST_ROTATION')

    assert frame.frame_id == 'TEST_ROTATION'
    assert Frame.as_frame('TEST_ROTATION').frame_id == 'TEST_ROTATION'


def test_rotation_pickle() -> None:
    """Pickling restores the angle, the axis, and the reference frame."""

    frame = Rotation(0.4, 'z', Frame.J2000)
    restored = pickle.loads(pickle.dumps(frame))

    assert isinstance(restored, Rotation)
    assert restored.angle == frame.angle
    assert restored.reference == frame.reference
    assert restored.transform_at_time(Scalar(0.)).matrix \
           == frame.transform_at_time(Scalar(0.)).matrix


def test_rotation_getstate_roundtrip() -> None:
    """The state captured by __getstate__ fully restores the object."""

    frame = Rotation(0.4, 'z', Frame.J2000)
    state = frame.__getstate__()

    copied = Frame.__new__(Rotation)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
