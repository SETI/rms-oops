##########################################################################################
# tests/host/test_host_index_rows.py: Host._index_rows streams, pairs and filters rows
##########################################################################################

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from oops.host import Host, HostIndexError, PdsHostError, WrongHostError, _index_rows
from tests.host._tables import write_table

VOLUMES = ['V1', 'V1', 'V1']


class _ByPosition(Host):
    """A host whose supplement lists the same files in the same order as its index."""

    NAME = 'by_position'

    @staticmethod
    def _detect_in_index(label: object) -> bool:
        return True


class _ByFile(_ByPosition):
    """A host whose supplement may list its files in another order."""

    NAME = 'by_file'
    MERGE_SUPPLEMENT_BY_FILE = True


class _FileNameKey(_ByPosition):
    """A host whose tables give the file in a FILE_NAME column alone."""

    NAME = 'file_name_key'
    INDEX_FILE_KEY = ('FILE_NAME',)
    SUPPLEMENT_FILE_KEY = ('FILE_NAME',)


class _NotMine(Host):
    """A host that recognizes no index."""

    NAME = 'not_mine'

    @staticmethod
    def _detect_in_index(label: object) -> bool:
        return False


class _ByRow(Host):
    """A host that recognizes an index only from a first row whose NUM is 1."""

    NAME = 'by_row'

    @staticmethod
    def _detect_in_row(row_dict: dict[str, Any]) -> bool:
        return bool(row_dict['NUM'] == '1')


def _index(directory: Path, files: str, *, stem: str = 'vol_index',
           volumes: list[str] | None = None, **columns: list[str]) -> Path:
    """Write an index that lists one file per letter of `files`; return its label."""

    return write_table(directory, stem,
                       {'VOLUME_ID': volumes or ['V1'] * len(files),
                        'FILE_SPECIFICATION_NAME': [f'DATA/{f}.LBL' for f in files],
                        **columns})


def _supplement(directory: Path, files: str, *, stem: str = 'vol_supplemental_index',
                volumes: list[str] | None = None, **columns: list[str]) -> Path:
    """Write a supplement that lists one file per letter of `files`; return its label."""

    return write_table(directory, stem,
                       {'VOLUME_ID': volumes or ['V1'] * len(files),
                        'FILE_SPECIFICATION_NAME': [f'DATA/{f}.IMG' for f in files],
                        **columns})


def _column(rows: Iterator[dict[str, Any]], name: str) -> list[Any]:
    """The value of one column in each row."""

    return [str(row[name]).strip() for row in rows]

##########################################################################################
# Streaming
##########################################################################################

def test_the_index_is_read_a_chunk_at_a_time(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """The table reader is asked for 100 rows at a time."""

    label = _index(tmp_path, 'AB')
    sizes = []
    iter_dicts = _index_rows.PdsTable.iter_dicts

    def recording(self: Any, chunk_rows: int, **kwargs: Any) -> Any:
        sizes.append(chunk_rows)
        return iter_dicts(self, chunk_rows, **kwargs)

    monkeypatch.setattr(_index_rows.PdsTable, 'iter_dicts', recording)
    list(Host._index_rows(label, supplement=None))

    assert sizes == [100]


def test_rows_are_read_as_the_iteration_reaches_them(tmp_path: Path) -> None:
    """A table file shorter than its label gives every row it holds, then the error."""

    label = write_table(tmp_path, 'vol_index', {'NUM': ['1', '2']}, label_rows=3)
    rows = Host._index_rows(label, supplement=None)

    assert str(next(rows)['NUM']) == '1'
    assert str(next(rows)['NUM']) == '2'
    with pytest.raises(HostIndexError, match='invalid index table .*vol_index.lbl'):
        next(rows)


def test_a_row_keeps_its_mask_flags(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    row = next(Host._index_rows(label, supplement=None))

    assert 'FILE_SPECIFICATION_NAME_mask' in row


def test_a_first_row_of_an_empty_table_is_none(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'NUM': ['1']}, label_rows=0)
    (tmp_path / 'vol_index.tab').write_bytes(b'')
    table = _index_rows.open_table(FCPath(label))

    assert _index_rows.first_row(table, FCPath(label)) is None

##########################################################################################
# The host's claim on the index
##########################################################################################

def test_an_index_of_another_host_is_refused(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')

    with pytest.raises(WrongHostError, match='not a not_mine index: .*vol_index.lbl'):
        _NotMine._index_rows(label, supplement=None)


def test_a_host_can_claim_the_index_from_its_first_row(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'NUM': ['1', '2']})

    assert _column(_ByRow._index_rows(label, supplement=None), 'NUM') == ['1', '2']


def test_a_host_can_refuse_the_index_from_its_first_row(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'NUM': ['2', '1']})

    with pytest.raises(WrongHostError, match='not a by_row index'):
        _ByRow._index_rows(label, supplement=None)


def test_a_given_label_is_used_in_place_of_the_file(tmp_path: Path) -> None:
    """The label file no longer parses, so only the given label can describe the table."""

    label = _index(tmp_path, 'A')
    parsed = Host._parse_pds3(FCPath(label), {})
    label.write_text('ROWS = = 1')

    assert _column(Host._index_rows(label, supplement=None, label=parsed),
                   'VOLUME_ID') == ['V1']

##########################################################################################
# Finding the supplement
##########################################################################################

def test_the_default_supplement_is_the_sibling_file(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB')
    _supplement(tmp_path, 'AB', VAL=['a', 'b'])

    assert _column(_ByPosition._index_rows(label), 'VAL') == ['a', 'b']


def test_an_uppercase_index_name_gives_an_uppercase_supplement_name() -> None:
    path = Host._supplement_path(FCPath('/data/VOL_INDEX.LBL'))

    assert path.name == 'VOL_SUPPLEMENTAL_INDEX.LBL'


def test_a_name_without_index_gives_no_supplement_name() -> None:
    assert Host._supplement_path(FCPath('/data/catalog.lbl')) is None


def test_a_host_can_name_its_supplement_another_way(tmp_path: Path) -> None:
    class _OwnName(_ByPosition):
        @classmethod
        def _supplement_path(cls, index_path: FCPath) -> FCPath:
            return index_path.with_name('extra.lbl')

    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'A', stem='extra', VAL=['x'])

    assert _column(_OwnName._index_rows(label), 'VAL') == ['x']


def test_an_explicit_supplement_in_another_directory_is_read(tmp_path: Path) -> None:
    other = tmp_path / 'other'
    other.mkdir()
    label = _index(tmp_path, 'A')
    supplement = _supplement(other, 'A', stem='anything', VAL=['x'])

    assert _column(_ByPosition._index_rows(label, supplement=supplement),
                   'VAL') == ['x']


def test_no_supplement_reads_the_index_alone(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'A', VAL=['x'])

    assert 'VAL' not in next(_ByPosition._index_rows(label, supplement=None))


def test_a_missing_explicit_supplement_is_not_found(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')

    with pytest.raises(FileNotFoundError, match='nowhere.lbl'):
        _ByPosition._index_rows(label, supplement=tmp_path / 'nowhere.lbl')


def test_a_missing_optional_explicit_supplement_is_skipped(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    rows = _ByPosition._index_rows(label, supplement=tmp_path / 'nowhere.lbl',
                                   require_supplement=False)

    assert _column(rows, 'VOLUME_ID') == ['V1']


def test_a_supplement_with_invalid_syntax_is_a_pds_error(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    supplement = _supplement(tmp_path, 'A', VAL=['x'])
    supplement.write_text(supplement.read_text().replace('ROWS = 1',
                                                         'ROWS = 1\nNOTE = "abc'))

    with pytest.raises(PdsHostError, match='cannot parse the PDS3 label .*supplemental'):
        _ByPosition._index_rows(label)


def test_a_label_without_its_table_object_is_an_index_error(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    label.write_text(label.read_text().replace('^INDEX_TABLE', '^TABLE'))

    with pytest.raises(HostIndexError, match='invalid index table .*: the label lacks'):
        _ByPosition._index_rows(label, supplement=None)


def test_a_label_with_an_unreadable_table_is_an_index_error(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    label.write_text(label.read_text().replace('INTERCHANGE_FORMAT = ASCII',
                                               'INTERCHANGE_FORMAT = EBCDIC'))

    with pytest.raises(HostIndexError, match='invalid index table .*vol_index.lbl: '):
        _ByPosition._index_rows(label, supplement=None)

##########################################################################################
# Pairing by position
##########################################################################################

def test_rows_in_the_same_order_pair_by_position(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB', NUM=['1', '2'])
    _supplement(tmp_path, 'AB', VAL=['a', 'b'])

    assert _column(_ByPosition._index_rows(label), 'VAL') == ['a', 'b']


def test_rows_of_different_files_stop_the_pairing_by_position(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB')
    _supplement(tmp_path, 'AC', VAL=['a', 'c'])
    rows = _ByPosition._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match="index row 1 gives 'V1/DATA/B.LBL' but"):
        next(rows)


def test_a_different_volume_stops_the_pairing_by_position(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A', volumes=['V1'])
    _supplement(tmp_path, 'A', volumes=['V2'], VAL=['a'])

    with pytest.raises(HostIndexError, match="index row 0 gives 'V1/DATA/A.LBL'"):
        next(_ByPosition._index_rows(label))


def test_a_short_supplement_stops_the_pairing_by_position(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB')
    _supplement(tmp_path, 'A', VAL=['a'])
    rows = _ByPosition._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match='the supplement ends before index row 1'):
        next(rows)


def test_a_long_supplement_fails_after_the_last_index_row(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'AB', VAL=['a', 'b'])
    rows = _ByPosition._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match='the supplement has more rows than the'):
        next(rows)


def test_the_index_value_wins_where_both_tables_have_a_column(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A', VAL=['index'])
    _supplement(tmp_path, 'A', VAL=['index-ish'])

    assert _column(_ByPosition._index_rows(label), 'VAL') == ['index']


def test_a_host_can_name_its_own_file_key(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'FILE_NAME': ['/V1/DATA/A.LBL']})
    write_table(tmp_path, 'vol_supplemental_index', {'FILE_NAME': ['V1/data/a.qub'],
                                                     'VAL': ['x']})

    assert _column(_FileNameKey._index_rows(label), 'VAL') == ['x']

##########################################################################################
# Pairing by file
##########################################################################################

def test_rows_out_of_order_pair_by_file(tmp_path: Path) -> None:
    label = _index(tmp_path, 'ABC')
    _supplement(tmp_path, 'CAB', VAL=['c', 'a', 'b'])

    assert _column(_ByFile._index_rows(label), 'VAL') == ['a', 'b', 'c']


def test_the_volume_tells_apart_rows_that_give_one_file(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AA', volumes=['V1', 'V2'])
    _supplement(tmp_path, 'AA', volumes=['V2', 'V1'], VAL=['two', 'one'])

    assert _column(_ByFile._index_rows(label), 'VAL') == ['one', 'two']


def test_a_file_the_supplement_lacks_fails_at_its_index_row(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB')
    _supplement(tmp_path, 'A', VAL=['a'])
    rows = _ByFile._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match="does not list the file of index row 1, "
                                             "'V1/DATA/B.LBL'"):
        next(rows)


def test_a_file_the_supplement_repeats_fails_at_the_repeat(tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB')
    _supplement(tmp_path, 'BBA', VAL=['b', 'b', 'a'])

    with pytest.raises(HostIndexError, match="gives 'V1/DATA/B.IMG' a second time, at "
                                             "row 1"):
        next(_ByFile._index_rows(label))


def test_files_the_index_lacks_fail_after_the_last_index_row(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'BA', VAL=['b', 'a'])
    rows = _ByFile._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match=r"lists files that the index does not, or "
                                             r"lists them more times, such as "
                                             r"\['V1/DATA/B.IMG'\]"):
        next(rows)


def test_unread_supplement_rows_fail_after_the_last_index_row(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'AB', VAL=['a', 'b'])
    rows = _ByFile._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match=r"such as \['V1/DATA/B.IMG'\]"):
        next(rows)

##########################################################################################
# Checking each pair
##########################################################################################

def test_the_default_check_compares_the_shared_columns(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A', TARGET=['SATURN'])
    _supplement(tmp_path, 'A', TARGET=['TITAN'])

    with pytest.raises(HostIndexError, match="TARGET value mismatch: 'SATURN', 'TITAN'\n"
                                             "index row 0 does not match"):
        next(_ByPosition._index_rows(label))


def test_the_default_check_leaves_out_the_file_key_columns(tmp_path: Path) -> None:
    """The two tables give the same product by its label and by its image file."""

    label = _index(tmp_path, 'A', TARGET=['SATURN'])
    _supplement(tmp_path, 'A', TARGET=['saturn'])

    assert _column(_ByPosition._index_rows(label), 'TARGET') == ['SATURN']


def test_tables_with_no_shared_column_are_refused(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'FILE_NAME': ['A.LBL']})
    write_table(tmp_path, 'vol_supplemental_index', {'NAME': ['A.LBL']})

    with pytest.raises(HostIndexError, match='index files have no columns in common'):
        _FileNameKey._index_rows(label)


def test_a_check_of_one_named_column(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A', TARGET=['SATURN'], NOTE=['x'])
    _supplement(tmp_path, 'A', TARGET=['SATURN'], NOTE=['y'])

    assert _column(_ByPosition._index_rows(label, validate='TARGET'), 'NOTE') == ['x']


def test_a_check_of_a_list_of_columns(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A', TARGET=['SATURN'], NOTE=['x'])
    _supplement(tmp_path, 'A', TARGET=['SATURN'], NOTE=['y'])

    with pytest.raises(HostIndexError, match="NOTE value mismatch: 'x', 'y'"):
        next(_ByPosition._index_rows(label, validate=['TARGET', 'NOTE']))


def test_a_check_of_numbers_compares_their_values(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index',
                        {'FILE_NAME': ['A.LBL'], 'NUM': ['1']}, integers=('NUM',))
    write_table(tmp_path, 'vol_supplemental_index',
                {'FILE_NAME': ['A.LBL'], 'NUM': ['2']}, integers=('NUM',))

    with pytest.raises(HostIndexError, match='NUM value mismatch: 1, 2'):
        next(_FileNameKey._index_rows(label))


def test_an_empty_check_compares_only_the_files(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A', TARGET=['SATURN'])
    _supplement(tmp_path, 'A', TARGET=['TITAN'])

    assert _column(_ByPosition._index_rows(label, validate=()), 'TARGET') == ['SATURN']


def test_a_check_function_that_returns_false_stops_the_pairing(tmp_path: Path) -> None:
    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'A')

    with pytest.raises(HostIndexError, match='^index row 0 does not match its '
                                             'supplement row'):
        next(_ByPosition._index_rows(label, validate=lambda index, supp: False))


def test_a_check_function_that_raises_stops_the_pairing(tmp_path: Path) -> None:
    def check(index_row: dict[str, Any], supplement_row: dict[str, Any]) -> bool:
        raise KeyError('NOPE')

    label = _index(tmp_path, 'A')
    _supplement(tmp_path, 'A')

    with pytest.raises(HostIndexError, match="'NOPE'\nindex row 0 does not match"):
        next(_ByPosition._index_rows(label, validate=check))

##########################################################################################
# Filtering
##########################################################################################

@pytest.mark.parametrize(('filter_', 'expected'),
                         [({'NUM': 2}, ['2']),
                          ({'NUM': {1, 3}}, ['1', '3']),
                          ({'NUM': [3]}, ['3']),
                          ({'NUM': (2, 3)}, ['2', '3']),
                          ({'NUM': (1, 2, 3)}, ['1', '2', '3'])],
                         ids=['value', 'set', 'list', 'range', 'long-tuple'])
def test_a_filter_passes_the_rows_it_selects(tmp_path: Path, filter_: dict[str, Any],
                                             expected: list[str]) -> None:
    label = write_table(tmp_path, 'vol_index', {'NUM': ['1', '2', '3']},
                        integers=('NUM',))

    assert _column(Host._index_rows(label, supplement=None, filter=filter_),
                   'NUM') == expected


def test_a_filter_column_that_the_rows_lack_raises_at_the_first_row(
        tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'NUM': ['1']})
    rows = Host._index_rows(label, supplement=None, filter={'OTHER': 1})

    with pytest.raises(KeyError, match='OTHER'):
        next(rows)


def test_given_rows_are_filtered_without_reading_the_tables(tmp_path: Path) -> None:
    rows = iter([{'NUM': 1}, {'NUM': 2}])
    result = _NotMine._index_rows(tmp_path / 'missing.lbl', row_dicts=rows,
                                  filter={'NUM': 2})

    assert list(result) == [{'NUM': 2}]


def test_a_filtered_merged_row_is_tested_on_the_supplement_columns(
        tmp_path: Path) -> None:
    label = _index(tmp_path, 'AB')
    _supplement(tmp_path, 'AB', VAL=['a', 'b'])

    assert _column(_ByFile._index_rows(label, filter={'VAL': 'b'}), 'VAL') == ['b']

##########################################################################################
# Comparing files
##########################################################################################

@pytest.mark.parametrize(('path', 'key'),
                         [('./data/a.lbl', 'DATA/A'),
                          ('/DATA//A.QUB', 'DATA/A'),
                          ('DATA/README', 'DATA/README'),
                          ('', '')],
                         ids=['dot-and-extension', 'slashes', 'no-extension', 'empty'])
def test_a_path_key_ignores_case_slashes_and_extension(path: str, key: str) -> None:
    assert _index_rows.path_key(path) == key

##########################################################################################
