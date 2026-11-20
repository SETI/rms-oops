##########################################################################################
# tests/host/test_host_support.py: the argument checks and index paths of Host
##########################################################################################

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from filecache import FCPath
from pdsparser import Pds3Label

from oops         import Frame, OopsSpiceError, OopsTypeError, OopsValueError
from oops.cadence import SnapCadence
from oops.frame   import SpiceErrorFrame
from oops.host    import PDS3_NULL_TEXTS, Host, is_pds3_null


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
# A frame that SPICE cannot supply, under the overrides that evaluate the frame
##########################################################################################

OWN_CADENCE = SnapCadence(100., 10.)        # midtime 105


def _spice_error_frame() -> SpiceErrorFrame:
    try:
        raise RuntimeError('SPICE(CKINSUFFDATA) -- no pointing')
    except RuntimeError as err:
        return SpiceErrorFrame(err, description='no test pointing')


def test_a_spice_error_frame_still_raises_when_used_after_a_tracker() -> None:
    params = {'cadence': OWN_CADENCE, 'frame': _spice_error_frame(), 'path': 'SSB',
              'target': 'NONE'}
    Host._apply_overrides(params, None, tracker='start', navigation=(0., 0.))
    with pytest.raises(OopsSpiceError, match='no test pointing'):
        Frame.as_frame(params['frame']).wrt(Frame.J2000).transform_at_time(105.)


def test_a_spice_error_frame_still_raises_when_used_after_a_parallel() -> None:
    params = {'cadence': OWN_CADENCE, 'frame': _spice_error_frame(), 'path': 'SSB'}
    Host._apply_overrides(params, None, parallel=SimpleNamespace(frame=Frame.J2000))
    with pytest.raises(OopsSpiceError, match='no test pointing'):
        Frame.as_frame(params['frame']).wrt(Frame.J2000).transform_at_time(105.)


def test_a_parallel_frame_that_spice_cannot_supply_raises_at_once() -> None:
    params = {'cadence': OWN_CADENCE, 'frame': Frame.J2000, 'path': 'SSB'}
    parallel = SimpleNamespace(frame=_spice_error_frame())
    with pytest.raises(OopsSpiceError, match='no test pointing'):
        Host._apply_overrides(params, None, parallel=parallel)

##########################################################################################
# PDS3 placeholder text
##########################################################################################

@pytest.mark.parametrize('value', ['UNK', 'unknown', ' N/A ', 'NULL', 'Not Applicable',
                                   '---', '', '  ', None, ['UNK', 'N/A'], ('NULL',)])
def test_a_placeholder_is_a_pds3_null(value: Any) -> None:
    assert is_pds3_null(value)


@pytest.mark.parametrize('value', ['UNKN', 'JUPITER', 0, -999., [], ['UNK', 'IO'],
                                   'N/A/B'])
def test_a_value_is_not_a_pds3_null(value: Any) -> None:
    assert not is_pds3_null(value)


def test_the_placeholder_texts_are_published() -> None:
    assert PDS3_NULL_TEXTS == frozenset({'UNK', 'UNKNOWN', 'N/A', 'NULL',
                                         'NOT APPLICABLE', '---', ''})

##########################################################################################
# The file of an index row
##########################################################################################

def test_the_file_of_a_row_is_in_its_volume() -> None:
    row = {'VOLUME_ID': 'COISS_2001 ', 'FILE_SPECIFICATION_NAME': ' data/N1.LBL'}
    assert Host._index_file_path(FCPath('/h/volumes/COISS_2xxx'), row) == \
        FCPath('/h/volumes/COISS_2xxx/COISS_2001/data/N1.LBL')


def test_the_file_of_a_row_can_come_from_other_columns() -> None:
    row = {'VOLUME_NAME': 'VGISS_6109', 'PATH': 'DATA/C1.IMG'}
    assert Host._index_file_path(FCPath('/v'), row, volume_column='VOLUME_NAME',
                                 file_column='PATH') == FCPath('/v/VGISS_6109/DATA/C1.IMG')

##########################################################################################
# Reading an index label
##########################################################################################

class _Reader(Host):
    """A host that reads an index label."""

    NAME = 'reader'


def test_the_index_label_is_parsed(tmp_path: Path) -> None:
    filepath = tmp_path / 'VOL_0001_index.lbl'
    filepath.write_text('PDS_VERSION_ID = PDS3\nRECORD_TYPE = FIXED_LENGTH\nEND\n')
    label = _Reader._read_index_label(filepath)
    assert label['RECORD_TYPE'] == 'FIXED_LENGTH'


def test_a_given_index_label_is_returned_without_reading_the_file(tmp_path: Path) -> None:
    label = Pds3Label('PDS_VERSION_ID = PDS3\nEND\n')
    assert _Reader._read_index_label(tmp_path / 'missing.lbl', label=label) is label

##########################################################################################
