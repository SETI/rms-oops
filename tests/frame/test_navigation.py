##########################################################################################
# oops/frame/navigation.py: Subclass Navigation of class Frame
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame, Navigation, SpiceFrame
from oops.path  import SpicePath


def test_navigation(core_kernels):
    _ = SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')
    time = Scalar(1.e8)

    # Two small rotations about perpendicular axes tilt the Z-axis by their
    # root sum of squares
    (ay, ax) = (1.e-3, 2.e-3)
    nav = Navigation((ay, ax), mars, frame_id='+')
    assert nav.frame_id == 'IAU_MARS_NAV'
    assert nav.nparams == 2
    assert isinstance(type(nav).link, property)
    assert nav.link is None

    zaxis = nav.transform_at_time(time).rotate(Vector3.ZAXIS)
    assert (zaxis.sep(Vector3.ZAXIS).vals
            == pytest.approx(np.sqrt(ay**2 + ax**2), abs=0.5e-8))

    # A third angle rotates about the Z-axis, leaving the Z-axis untouched
    nav3 = Navigation((0., 0., 0.1), mars, frame_id='nav3')
    assert nav3.nparams == 3
    assert nav3.transform_at_time(time).rotate(Vector3.ZAXIS) == Vector3.ZAXIS
    xaxis = nav3.transform_at_time(time).rotate(Vector3.XAXIS)
    assert xaxis.sep(Vector3.XAXIS).vals == pytest.approx(0.1, abs=0.5e-12)

    # A linked Navigation tracks the angles of the object it is linked to
    linked = Navigation(nav, mars, frame_id='nav_linked')
    assert linked.link is nav
    assert linked.nparams == 2

    linked.set_params(np.array([5.e-3, 6.e-3]))
    nav.refresh()
    assert tuple(nav.angles) == (5.e-3, 6.e-3)
    assert tuple(linked.angles) == (5.e-3, 6.e-3)

    # Freezing severs the link but preserves the angles
    frozen = Navigation(linked, mars, frame_id='nav_frozen', freeze=True)
    assert frozen.is_frozen
    assert tuple(frozen.angles) == (5.e-3, 6.e-3)


def test_navigation_takes_two_or_three_angles(core_kernels) -> None:
    """Two angles rotate about y and x; a third adds a rotation about z."""

    assert Navigation((0.01, 0.02), Frame.J2000).angles == (0.01, 0.02)
    assert Navigation((0.01, 0.02, 0.03), Frame.J2000).angles == (0.01, 0.02, 0.03)


@pytest.mark.parametrize('angles', [(0.01,), (0.01, 0.02, 0.03, 0.04), ()],
                         ids=['one', 'four', 'none'])
def test_navigation_rejects_any_other_count(angles: tuple, core_kernels) -> None:
    """Anything but two or three angles raises ValueError."""

    with pytest.raises(ValueError, match='two or three Navigation angles'):
        Navigation(angles, Frame.J2000)


def test_navigation_rejects_an_unregistered_reference(core_kernels) -> None:
    """A reference ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        Navigation((0.01, 0.02), 'NOT_A_REGISTERED_FRAME')


def test_navigation_of_zero_angles_is_the_identity(core_kernels) -> None:
    """No rotation leaves every vector where it was."""

    frame = Navigation((0., 0.), Frame.J2000)
    matrix = frame.transform_at_time(Scalar(0.)).matrix

    assert matrix.rotate(Vector3.XAXIS) == Vector3.XAXIS
    assert matrix.rotate(Vector3.YAXIS) == Vector3.YAXIS
    assert matrix.rotate(Vector3.ZAXIS) == Vector3.ZAXIS


def test_navigation_is_fixed_in_time(core_kernels) -> None:
    """A Navigation describes a fixed offset, so it does not depend on time."""

    frame = Navigation((0.01, 0.02), Frame.J2000)
    transform = frame.transform_at_time(Scalar(0.))

    assert transform.is_fixed
    assert frame.transform_at_time(Scalar(1.e9)).matrix == transform.matrix


def test_navigation_is_fittable(core_kernels) -> None:
    """The rotation angles are the fittable parameters."""

    frame = Navigation((0.01, 0.02), Frame.J2000)

    assert frame.params == (0.01, 0.02)
    assert frame.nparams == 2
    assert not frame.is_frozen


def test_navigation_set_params_changes_the_angles(core_kernels) -> None:
    """Fitting the frame re-points it."""

    frame = Navigation((0.01, 0.02), Frame.J2000)
    frame.set_params((0.05, 0.06))

    assert frame.angles == (0.05, 0.06)


def test_navigation_freeze_blocks_fitting(core_kernels) -> None:
    """freeze=True returns an object that can no longer be fitted."""

    frozen = Navigation((0.01, 0.02), Frame.J2000, freeze=True)

    assert frozen.is_frozen
    assert frozen.angles == (0.01, 0.02)


def test_navigation_tracks_a_linked_navigation(core_kernels) -> None:
    """A linked Navigation always matches the angles of the object it is linked to."""

    source = Navigation((0.01, 0.02), Frame.J2000, frame_id='nav_source')
    linked = Navigation(source, Frame.J2000, frame_id='nav_follower')

    assert linked.link is source
    assert linked.angles == (0.01, 0.02)

    source.set_params((0.05, 0.06))
    linked.refresh()
    assert linked.angles == (0.05, 0.06)


def test_navigation_of_an_unlinked_object_has_no_link(core_kernels) -> None:
    """An object given its own angles is not linked to anything."""

    assert Navigation((0.01, 0.02), Frame.J2000).link is None


def test_navigation_auto_generated_frame_id(core_kernels) -> None:
    """A frame_id of "+" appends "_NAV" to the reference frame's ID."""

    assert Navigation((0., 0.), Frame.J2000, frame_id='+').frame_id == 'J2000_NAV'


def test_navigation_pickle(core_kernels) -> None:
    """Pickling restores the angles and the reference frame."""

    frame = Navigation((0.01, 0.02), Frame.J2000)
    restored = pickle.loads(pickle.dumps(frame))

    assert isinstance(restored, Navigation)
    assert restored.angles == frame.angles
    assert restored.transform_at_time(Scalar(0.)).matrix \
           == frame.transform_at_time(Scalar(0.)).matrix


def test_navigation_getstate_roundtrip(core_kernels) -> None:
    """The state captured by __getstate__ fully restores the object."""

    frame = Navigation((0.01, 0.02), Frame.J2000)
    state = frame.__getstate__()

    copied = Frame.__new__(Navigation)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
