##########################################################################################
# tests/frame/test_frame_registry.py: the Frame registry and the Frame base class
##########################################################################################


import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame, Rotation, SpiceFrame
from oops.path  import SpicePath


@pytest.fixture
def mars_frame(core_kernels) -> SpiceFrame:
    """The IAU_MARS body-fixed frame, with the Mars path registered alongside it."""

    SpicePath('MARS', 'SSB')

    return SpiceFrame('IAU_MARS', 'J2000')


def test_frame_id_exists_after_registration(mars_frame: SpiceFrame) -> None:
    """A registered frame ID is present in the registry."""

    assert Frame.frame_id_exists('IAU_MARS')


def test_frame_id_exists_is_false_for_an_unknown_id(mars_frame: SpiceFrame) -> None:
    """An ID that was never registered is absent from the registry."""

    assert not Frame.frame_id_exists('NOT_A_REGISTERED_FRAME')


def test_as_frame_returns_a_frame_unchanged(mars_frame: SpiceFrame) -> None:
    """A Frame is already a Frame, so it is returned as it is."""

    assert Frame.as_frame(mars_frame) is mars_frame


def test_as_frame_converts_an_id(mars_frame: SpiceFrame) -> None:
    """A registered ID string names its Frame."""

    assert Frame.as_frame('IAU_MARS').frame_id == 'IAU_MARS'


def test_as_frame_rejects_an_unregistered_id(mars_frame: SpiceFrame) -> None:
    """An ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        Frame.as_frame('NOT_A_REGISTERED_FRAME')


def test_as_wayframe_gives_the_canonical_frame(mars_frame: SpiceFrame) -> None:
    """The wayframe is the canonical definition, used as a key for indexing."""

    assert Frame.as_wayframe(mars_frame) is mars_frame.wayframe
    assert Frame.as_wayframe('IAU_MARS') is mars_frame.wayframe


def test_as_wayframe_rejects_an_unregistered_id(mars_frame: SpiceFrame) -> None:
    """An ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        Frame.as_wayframe('NOT_A_REGISTERED_FRAME')


def test_as_primary_frame_gives_the_primary_definition(mars_frame: SpiceFrame) -> None:
    """The primary frame is the first object registered under the ID."""

    assert Frame.as_primary_frame('IAU_MARS') is mars_frame.primary


def test_a_registered_frame_reports_its_id(mars_frame: SpiceFrame) -> None:
    """A registered frame carries the ID it was registered under."""

    frame = Rotation(0.3, 'z', mars_frame, frame_id='TEST_REGISTERED_ROTATION')

    assert frame.is_registered
    assert frame.frame_id == 'TEST_REGISTERED_ROTATION'
    assert frame.string_id == 'TEST_REGISTERED_ROTATION'


def test_an_unregistered_frame_has_no_id(mars_frame: SpiceFrame) -> None:
    """An unregistered frame gets a unique string derived from its identity."""

    frame = Rotation(0.3, 'z', mars_frame)

    assert not frame.is_registered
    assert frame.frame_id is None
    assert frame.string_id.startswith('#')


def test_j2000_is_inertial() -> None:
    """The J2000 frame does not rotate."""

    assert Frame.J2000.is_inertial


def test_a_body_fixed_frame_is_not_inertial(mars_frame: SpiceFrame) -> None:
    """A frame tied to a rotating body is not inertial."""

    assert not mars_frame.is_inertial


def test_an_inertial_frame_has_no_origin() -> None:
    """The origin Path is None for an inertial frame."""

    assert Frame.J2000.origin is None


def test_a_rotating_frame_has_an_origin(mars_frame: SpiceFrame) -> None:
    """A rotating frame names the path at its center of rotation."""

    assert mars_frame.origin is not None


def test_wrt_returns_a_frame_relative_to_the_reference(mars_frame: SpiceFrame) -> None:
    """wrt() connects this Frame to another one."""

    relative = mars_frame.wrt(Frame.J2000)

    assert isinstance(relative, Frame)
    assert relative.reference == Frame.J2000.wayframe


def test_wrt_j2000_is_wrt_the_j2000_frame(mars_frame: SpiceFrame) -> None:
    """The property is the same connection as wrt(J2000)."""

    assert mars_frame.wrt_j2000.reference == mars_frame.wrt(Frame.J2000).reference


def test_wrt_itself_is_the_identity(mars_frame: SpiceFrame) -> None:
    """A frame relative to itself does not rotate anything."""

    transform = mars_frame.wrt(mars_frame).transform_at_time(Scalar(1.e8))

    assert transform.rotate(Vector3.XAXIS) == Vector3.XAXIS


def test_show_names_the_frame(mars_frame: SpiceFrame) -> None:
    """The description names this Frame and the Frames it is built from."""

    text = mars_frame.show(2)

    assert 'IAU_MARS' in text


def test_show_of_a_derived_frame_names_its_reference(mars_frame: SpiceFrame) -> None:
    """A frame built on another names that one too."""

    text = Rotation(0.3, 'z', mars_frame, frame_id='TEST_SHOWN_ROTATION').show(3)

    assert 'IAU_MARS' in text


def test_repr_and_str_agree(mars_frame: SpiceFrame) -> None:
    """The two string forms of a Frame are the same."""

    assert repr(mars_frame) == str(mars_frame)


def test_reset_caches_empties_the_registry(mars_frame: SpiceFrame) -> None:
    """Resetting the caches forgets every registered frame but the built-in ones."""

    assert Frame.frame_id_exists('IAU_MARS')
    Frame._reset_caches()

    assert not Frame.frame_id_exists('IAU_MARS')
    assert Frame.frame_id_exists('J2000')


def test_pickle_quickframe_details_can_be_turned_on(mars_frame: SpiceFrame) -> None:
    """The flag controls whether QuickFrame tabulations are pickled with the Frame."""

    original = mars_frame.pickle_quickframe_details
    try:
        mars_frame.pickle_quickframe_details = True
        assert mars_frame.pickle_quickframe_details

        mars_frame.pickle_quickframe_details = False
        assert not mars_frame.pickle_quickframe_details
    finally:
        mars_frame.pickle_quickframe_details = original

##########################################################################################
