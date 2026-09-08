##########################################################################################
# tests/test_spice_support.py
##########################################################################################

from collections.abc import Iterator

import pytest

import oops.spice_support as spice_support
from oops.frame import Frame, SpiceFrame
from oops.path  import FixedPath, Path, SpicePath


@pytest.fixture(autouse=True)
def _fresh_translation_tables(core_kernels: None) -> Iterator[None]:
    """Reset the translation tables around each test.

    The tables are module-level dictionaries, so a name one test registers would
    otherwise be visible to every test that runs afterward.
    """

    spice_support.initialize()

    yield

    spice_support.initialize()
    Path._reset_caches()
    Frame._reset_caches()


def test_initialize_keeps_only_the_built_in_entries() -> None:
    """Only J2000 and the solar system barycenter survive a reset."""

    spice_support.initialize()

    assert set(spice_support.FRAME_TRANSLATION.values()) == {'J2000'}
    assert set(spice_support.PATH_TRANSLATION.values()) == {'SSB'}


def test_initialize_forgets_a_registered_path_name() -> None:
    """A path name registered since the last call is forgotten."""

    spice_support.PATH_TRANSLATION['MARS'] = 'MARS'
    spice_support.initialize()

    assert 'MARS' not in spice_support.PATH_TRANSLATION


def test_initialize_forgets_a_registered_frame_name() -> None:
    """A frame name registered since the last call is forgotten."""

    spice_support.FRAME_TRANSLATION['IAU_MARS'] = 'IAU_MARS'
    spice_support.initialize()

    assert 'IAU_MARS' not in spice_support.FRAME_TRANSLATION


def test_body_id_and_name_from_a_name() -> None:
    """A recognized body name yields its SPICE ID and canonical name."""

    (body_id, name) = spice_support.body_id_and_name('MARS')

    assert body_id == 499
    assert name == 'MARS'


def test_body_id_and_name_from_an_id() -> None:
    """A SPICE body ID yields the same pair as its name."""

    assert spice_support.body_id_and_name(499) == spice_support.body_id_and_name('MARS')


def test_body_id_and_name_of_the_barycenter() -> None:
    """The solar system barycenter is present in the table from the start."""

    (body_id, name) = spice_support.body_id_and_name('SSB')

    assert body_id == 0
    assert name == 'SSB'


def test_body_id_and_name_is_case_insensitive() -> None:
    """The SPICE Toolkit resolves a body name regardless of its case."""

    assert spice_support.body_id_and_name('mars') == spice_support.body_id_and_name('MARS')


def test_body_id_and_name_rejects_an_unknown_name() -> None:
    """A name the Toolkit does not recognize raises LookupError."""

    with pytest.raises(LookupError):
        spice_support.body_id_and_name('NOT_A_BODY')


def test_body_id_and_name_rejects_a_non_name() -> None:
    """An argument that is neither a name nor an integer raises LookupError."""

    with pytest.raises(LookupError):
        spice_support.body_id_and_name(3.14159)


def test_body_id_and_name_of_an_unnamed_id() -> None:
    """A body with no name in the Toolkit is given the string form of its ID."""

    (body_id, name) = spice_support.body_id_and_name(-999999)

    assert body_id == -999999
    assert name == '-999999'


def test_body_id_and_name_resolves_a_registered_path() -> None:
    """A name already in the path translation table is resolved through its Path."""

    path = SpicePath('MARS', 'SSB')
    spice_support.PATH_TRANSLATION['MARS'] = path.path_id

    assert spice_support.body_id_and_name('MARS') == (499, 'MARS')


def test_body_id_and_name_rejects_a_path_that_is_not_a_spicepath() -> None:
    """A name resolving to a Path that is not a SpicePath raises TypeError."""

    FixedPath((0., 0., 0.), Path.SSB, path_id='NOT_A_SPICE_PATH')
    spice_support.PATH_TRANSLATION['NOT_A_SPICE_PATH'] = 'NOT_A_SPICE_PATH'

    with pytest.raises(TypeError):
        spice_support.body_id_and_name('NOT_A_SPICE_PATH')


def test_frame_id_and_name_from_a_frame_name() -> None:
    """A recognized frame name yields its SPICE ID and canonical name."""

    (frame_id, name) = spice_support.frame_id_and_name('IAU_MARS')

    assert name == 'IAU_MARS'
    assert isinstance(frame_id, int)


def test_frame_id_and_name_from_a_frame_id() -> None:
    """A SPICE frame ID yields the same pair as its name."""

    (frame_id, _) = spice_support.frame_id_and_name('IAU_MARS')

    assert spice_support.frame_id_and_name(frame_id) \
           == spice_support.frame_id_and_name('IAU_MARS')


def test_frame_id_and_name_resolves_a_registered_frame() -> None:
    """A name already in the frame translation table is resolved through its Frame."""

    frame = SpiceFrame('IAU_MARS')
    spice_support.FRAME_TRANSLATION['IAU_MARS'] = frame.frame_id

    assert spice_support.frame_id_and_name('IAU_MARS')[1] == 'IAU_MARS'


def test_frame_id_and_name_of_j2000() -> None:
    """J2000 is present in the table from the start."""

    (_, name) = spice_support.frame_id_and_name('J2000')

    assert name == 'J2000'


def test_frame_id_and_name_falls_back_to_the_body_frame() -> None:
    """An argument naming a body yields the frame associated with that body."""

    assert spice_support.frame_id_and_name('MARS') \
           == spice_support.frame_id_and_name('IAU_MARS')


def test_frame_id_and_name_rejects_an_unknown_name() -> None:
    """A name that is neither a frame nor a body raises LookupError."""

    with pytest.raises(LookupError):
        spice_support.frame_id_and_name('NOT_A_FRAME')


def test_frame_id_and_name_rejects_a_non_name() -> None:
    """An argument that is neither a string nor an integer raises LookupError."""

    with pytest.raises(LookupError):
        spice_support.frame_id_and_name(3.14159)


def test_load_leap_seconds_is_idempotent() -> None:
    """The leap seconds kernel is loaded only if it was not already loaded."""

    spice_support.load_leap_seconds()
    spice_support.load_leap_seconds()

    assert spice_support.LSK_LOADED

def test_frame_id_and_name_from_a_body_id() -> None:
    """A body ID that is not a frame ID resolves to that body's rotation frame."""

    assert spice_support.frame_id_and_name(499) == (499, 'IAU_MARS')


def test_frame_id_and_name_rejects_a_body_id_with_no_frame() -> None:
    """A spacecraft has no rotation frame defined by the planetary constants.

    New Horizons is a body the Toolkit knows, but no test here furnishes a frame kernel
    for it, so it has no rotation frame.
    """

    with pytest.raises(LookupError, match='frame for body -98 is undefined'):
        spice_support.frame_id_and_name(-98)


def test_frame_id_and_name_rejects_a_frame_whose_body_has_no_pole() -> None:
    """A frame is defined only if the planetary constants give its body a pole."""

    with pytest.raises(LookupError, match='frame "EARTH_BARYCENTER" is undefined'):
        spice_support.frame_id_and_name('EARTH_BARYCENTER')


def test_frame_id_and_name_rejects_a_body_name_with_no_frame() -> None:
    """A spacecraft named as a string is refused for the same reason."""

    with pytest.raises(LookupError, match='frame for body "NEW HORIZONS" is undefined'):
        spice_support.frame_id_and_name('NEW HORIZONS')

##########################################################################################
