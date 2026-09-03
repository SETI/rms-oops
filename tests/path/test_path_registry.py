##########################################################################################
# tests/path/test_path_registry.py: the Path registry and the Path base class
##########################################################################################

import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame
from oops.path  import FixedPath, Path, SpicePath


@pytest.fixture
def mars_path(core_kernels) -> SpicePath:
    """The path of Mars relative to the solar system barycenter."""

    return SpicePath('MARS', 'SSB')


def test_path_id_exists_after_registration(mars_path: SpicePath) -> None:
    """A registered path ID is present in the registry."""

    assert Path.path_id_exists('MARS')


def test_path_id_exists_is_false_for_an_unknown_id(mars_path: SpicePath) -> None:
    """An ID that was never registered is absent from the registry."""

    assert not Path.path_id_exists('NOT_A_REGISTERED_PATH')


def test_as_path_returns_a_path_unchanged(mars_path: SpicePath) -> None:
    """A Path is already a Path, so it is returned as it is."""

    assert Path.as_path(mars_path) is mars_path


def test_as_path_converts_an_id(mars_path: SpicePath) -> None:
    """A registered ID string names its Path."""

    assert Path.as_path('MARS').path_id == 'MARS'


def test_as_path_rejects_an_unregistered_id(mars_path: SpicePath) -> None:
    """An ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        Path.as_path('NOT_A_REGISTERED_PATH')


def test_as_waypoint_gives_the_canonical_path(mars_path: SpicePath) -> None:
    """The waypoint is the canonical definition, used as a key for indexing."""

    assert Path.as_waypoint(mars_path) is mars_path.waypoint
    assert Path.as_waypoint('MARS') is mars_path.waypoint


def test_as_waypoint_rejects_an_unregistered_id(mars_path: SpicePath) -> None:
    """An ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        Path.as_waypoint('NOT_A_REGISTERED_PATH')


def test_as_primary_path_gives_the_primary_definition(mars_path: SpicePath) -> None:
    """The primary path is the first object registered under the ID."""

    assert Path.as_primary_path('MARS') is mars_path.primary


def test_a_registered_path_reports_its_id(mars_path: SpicePath) -> None:
    """A registered path carries the ID it was registered under."""

    path = FixedPath((1.e5, 0., 0.), Path.SSB, path_id='TEST_REGISTERED_PATH')

    assert path.is_registered
    assert path.path_id == 'TEST_REGISTERED_PATH'
    assert path.string_id == 'TEST_REGISTERED_PATH'


def test_an_unregistered_path_has_no_id(mars_path: SpicePath) -> None:
    """An unregistered path gets a unique string derived from its identity."""

    path = FixedPath((1.e5, 0., 0.), Path.SSB)

    assert not path.is_registered
    assert path.path_id is None
    assert path.string_id.startswith('#')


def test_the_barycenter_is_at_the_origin() -> None:
    """The SSB is the root of the path registry, so it is its own origin."""

    assert Path.SSB.path_id == 'SSB'
    assert Path.SSB.event_at_time(Scalar(0.)).pos == Vector3.ZERO


def test_a_path_reports_its_frame(mars_path: SpicePath) -> None:
    """The coordinates of a path are expressed in its frame."""

    assert mars_path.frame == Frame.J2000.wayframe


def test_a_simple_path_is_shapeless(mars_path: SpicePath) -> None:
    """A single body's path holds one position at a time."""

    assert mars_path.shape == ()


def test_wrt_returns_a_path_relative_to_the_origin(mars_path: SpicePath) -> None:
    """wrt() connects this Path to another one."""

    relative = mars_path.wrt(Path.SSB)

    assert isinstance(relative, Path)
    assert relative.origin == Path.SSB.waypoint


def test_wrt_itself_is_at_the_origin(mars_path: SpicePath) -> None:
    """A path relative to itself sits at zero."""

    assert mars_path.wrt(mars_path).event_at_time(Scalar(1.e8)).pos == Vector3.ZERO


def test_wrt_reverses_the_position(mars_path: SpicePath) -> None:
    """Mars relative to the SSB is the negative of the SSB relative to Mars."""

    time = Scalar(1.e8)
    outward = mars_path.wrt(Path.SSB).event_at_time(time).pos
    inward = Path.SSB.wrt(mars_path).event_at_time(time).pos

    assert outward == -inward


def test_show_names_the_path(mars_path: SpicePath) -> None:
    """The description names this Path and the Paths it is built from."""

    assert 'MARS' in mars_path.show(2)


def test_repr_and_str_agree(mars_path: SpicePath) -> None:
    """The two string forms of a Path are the same."""

    assert repr(mars_path) == str(mars_path)


def test_reset_caches_empties_the_registry(mars_path: SpicePath) -> None:
    """Resetting the caches forgets every registered path but the built-in ones."""

    assert Path.path_id_exists('MARS')
    Path._reset_caches()

    assert not Path.path_id_exists('MARS')
    assert Path.path_id_exists('SSB')


def test_photon_to_event_travels_at_the_speed_of_light(mars_path: SpicePath) -> None:
    """The photon leaves this path early enough to arrive at the given event."""

    arrival = Path.SSB.event_at_time(Scalar(1.e8))
    (departure, arrival_out) = mars_path.photon_to_event(arrival)

    assert departure.time < arrival_out.time
    assert arrival_out.arr_lt < 0.

##########################################################################################
