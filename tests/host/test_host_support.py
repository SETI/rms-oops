##########################################################################################
# tests/host/test_host_support.py: the argument checks and index paths of Host
##########################################################################################

from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath
from pdsparser import Pds3Label

from oops         import OopsTypeError, OopsValueError
from oops.cadence import SnapCadence
from oops.host    import Host, HostIndexError


class _Named(Host):
    """A host whose class name appears in messages."""

    NAME = 'named'


@pytest.mark.parametrize(('name', 'expected'),
                         [('VOL_0001_index.lbl', 'VOL_0001_supplemental_index.lbl'),
                          ('VOL_0001_INDEX.LBL', 'VOL_0001_SUPPLEMENTAL_INDEX.LBL')],
                         ids=['lower', 'upper'])
def test_the_supplement_is_named_after_the_index(name: str, expected: str) -> None:
    assert Host._supplement_path(FCPath('/data') / name) == FCPath('/data') / expected


def test_an_index_without_index_in_its_name_has_no_supplement() -> None:
    assert Host._supplement_path(FCPath('/data/table.lbl')) is None


@pytest.mark.parametrize(('index', 'expected'),
                         [('/h/metadata/GO_0xxx/GO_0017/GO_0017_index.lbl',
                           '/h/volumes/GO_0xxx'),
                          ('/h/METADATA/GO_0xxx/GO_0017/GO_0017_index.lbl',
                           '/h/volumes/GO_0xxx'),
                          ('/elsewhere/GO_0017_index.lbl', '/elsewhere')],
                         ids=['metadata', 'upper-case', 'elsewhere'])
def test_the_volumes_directory_follows_the_location_of_the_index(index: str,
                                                                 expected: str) -> None:
    assert Host._volumes_directory(index) == FCPath(expected)


def test_an_unexpected_keyword_raises_naming_the_class_and_method() -> None:
    with pytest.raises(OopsTypeError, match="unexpected keyword argument for "
                                            "_Named.from_index: 'supplementary'"):
        _Named._check_keywords('from_index', {'supplementary': 1})


def test_select_must_be_none_for_one_observation_per_row() -> None:
    with pytest.raises(OopsValueError, match='each row of an index is one observation; '
                                             'select must be None, not 0'):
        Host._check_select(0, unit='row of an index')


def test_select_with_an_unknown_name_raises() -> None:
    with pytest.raises(OopsValueError, match="unrecognized select name: 'VIS'"):
        Host._check_select(('IR', 'VIS'), names={'IR': 0})


def test_select_of_another_type_raises() -> None:
    with pytest.raises(OopsTypeError, match='invalid select: 1.5'):
        Host._check_select(1.5)


def test_a_valid_select_passes() -> None:
    assert Host._check_select((slice(None), 'ir', 0), names={'IR': 0}) is None

##########################################################################################
# The midtime of the pointing check after the overrides
##########################################################################################

OWN_CADENCE = SnapCadence(100., 10.)        # midtime 105
NO_OVERRIDES: dict[str, Any] = {'cadence': None, 'timeshift': None}


def test_without_overrides_the_midtime_is_the_cadences() -> None:
    assert Host._override_midtime(OWN_CADENCE, NO_OVERRIDES) == 105.


def test_a_cadence_override_gives_its_midtime() -> None:
    overrides = NO_OVERRIDES | {'cadence': SnapCadence(200., 4.)}
    assert Host._override_midtime(OWN_CADENCE, overrides) == 202.


@pytest.mark.parametrize('letters', ['T', 'pt'])
def test_a_timeshift_of_the_cadence_moves_the_midtime(letters: str) -> None:
    overrides = NO_OVERRIDES | {'timeshift': (3., letters)}
    assert Host._override_midtime(OWN_CADENCE, overrides) == 108.


def test_a_timeshift_of_the_frame_alone_leaves_the_midtime() -> None:
    overrides = NO_OVERRIDES | {'timeshift': (3., 'F')}
    assert Host._override_midtime(OWN_CADENCE, overrides) == 105.


def test_overrides_that_are_not_given_count_as_none() -> None:
    assert Host._override_midtime(OWN_CADENCE, {}) == 105.

##########################################################################################
# Reading an index label
##########################################################################################

class _MetadataReader(Host):
    """A host that reads only the metadata indexes."""

    NAME = 'metadata_reader'
    METADATA_INDEX_EXAMPLE = 'VOL_0001_index.lbl'


def test_the_index_label_is_parsed(tmp_path: Path) -> None:
    filepath = tmp_path / 'VOL_0001_index.lbl'
    filepath.write_text('PDS_VERSION_ID = PDS3\nRECORD_TYPE = FIXED_LENGTH\nEND\n')
    label = _MetadataReader._read_index_label(filepath)
    assert label['RECORD_TYPE'] == 'FIXED_LENGTH'


def test_a_given_index_label_is_returned_without_reading_the_file(tmp_path: Path) -> None:
    label = Pds3Label('PDS_VERSION_ID = PDS3\nEND\n')
    assert _MetadataReader._read_index_label(tmp_path / 'missing.lbl',
                                             label=label) is label


def test_a_volumes_own_index_is_refused_before_its_label_is_read(tmp_path: Path) -> None:
    with pytest.raises(HostIndexError, match='metadata_reader reads the Ring-Moon Systems '
                                             'Node metadata index, such as '
                                             '"VOL_0001_index.lbl", not a volume\'s own '
                                             'index: .*missing.lbl'):
        _MetadataReader._read_index_label(tmp_path / 'INDEX' / 'missing.lbl')

##########################################################################################
