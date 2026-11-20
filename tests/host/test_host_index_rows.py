##########################################################################################
# tests/host/test_host_index_rows.py: Host._index_rows reads, pairs and filters lazily
##########################################################################################

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oops      import OopsTypeError, OopsValueError
from oops.host import Host, HostIndexError, WrongHostError, _index_rows
from tests.host._tables import write_table


class _ByPosition(Host):
    """A host whose index and supplement list the same files in the same order."""

    NAME = 'by-position'
    MERGE_SUPPLEMENT_BY_FILE = False

    @staticmethod
    def _detect_in_index(label: object) -> bool:
        return True


class _ByFile(Host):
    """A host whose supplement may list the files in another order."""

    NAME = 'by-file'
    MERGE_SUPPLEMENT_BY_FILE = True

    @staticmethod
    def _detect_in_index(label: object) -> bool:
        return True


def _tables(directory: Path, index: dict[str, list[str]],
            supplement: dict[str, list[str]]) -> Path:
    """Write an index and its supplement; return the index label."""

    write_table(directory, 'vol_supplemental_index', supplement)
    return write_table(directory, 'vol_index', index)


def _files(names: str) -> list[str]:
    """One file name per letter, such as ["A.LBL", "B.LBL"] for "AB"."""

    return [name + '.LBL' for name in names]


def _pairs(rows: Iterator[dict[str, Any]]) -> list[tuple[Any, Any]]:
    """The NUM and VAL of each row."""

    return [(row['NUM'], row['VAL']) for row in rows]


##########################################################################################
# Pairing by position
##########################################################################################

def test_rows_in_the_same_order_pair_by_position(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NUM': ['1', '2']},
                    {'FILE_SPECIFICATION_NAME': _files('AB'), 'VAL': ['a', 'b']})

    assert _pairs(_ByPosition._index_rows(label)) == [('1', 'a'), ('2', 'b')]


def test_the_default_file_key_is_the_file_specification_name_alone(
        tmp_path: Path) -> None:
    """Both tables have VOLUME_ID, and the message gives the file without it."""

    label = _tables(tmp_path, {'VOLUME_ID': ['V', 'V'],
                               'FILE_SPECIFICATION_NAME': _files('AB')},
                    {'VOLUME_ID': ['V', 'V'],
                     'FILE_SPECIFICATION_NAME': _files('AC')})
    rows = _ByPosition._index_rows(label)

    next(rows)
    with pytest.raises(HostIndexError, match="index row 1 gives 'B.LBL' but"):
        next(rows)


def test_the_default_check_compares_the_volume_of_rows_that_give_one_file(
        tmp_path: Path) -> None:
    label = _tables(tmp_path,
                    {'VOLUME_ID': ['V1'], 'FILE_SPECIFICATION_NAME': _files('A')},
                    {'VOLUME_ID': ['V2'], 'FILE_SPECIFICATION_NAME': _files('A')})
    rows = _ByPosition._index_rows(label)

    with pytest.raises(HostIndexError, match="VOLUME_ID value mismatch: 'V1', 'V2'"):
        next(rows)


def test_pairing_by_position_raises_at_the_first_row_whose_files_differ(
        tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('ABC'),
                               'NUM': ['1', '2', '3']},
                    {'FILE_SPECIFICATION_NAME': _files('ACB'), 'VAL': ['a', 'c', 'b']})
    rows = _ByPosition._index_rows(label)

    assert next(rows)['VAL'] == 'a'
    with pytest.raises(HostIndexError, match=r"index row 1 gives 'B.LBL' but the "
                                             r"supplement row in the same position "
                                             r"gives 'C.LBL'"):
        next(rows)


def test_pairing_by_position_raises_where_the_supplement_ends(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NUM': ['1', '2']},
                    {'FILE_SPECIFICATION_NAME': _files('A'), 'VAL': ['a']})
    rows = _ByPosition._index_rows(label)

    assert next(rows)['VAL'] == 'a'
    with pytest.raises(HostIndexError, match='the supplement ends before index row 1'):
        next(rows)


def test_pairing_by_position_raises_after_the_index_for_extra_supplement_rows(
        tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('A'), 'NUM': ['1']},
                    {'FILE_SPECIFICATION_NAME': _files('AB'), 'VAL': ['a', 'b']})
    rows = _ByPosition._index_rows(label)

    assert next(rows)['VAL'] == 'a'
    with pytest.raises(HostIndexError, match='the supplement has more rows than the '
                                             'index'):
        next(rows)


def test_a_pair_that_fails_validation_raises_at_its_row(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NAME': ['x', 'y']},
                    {'FILE_SPECIFICATION_NAME': _files('AB'), 'NAME': ['x', 'z']})
    rows = _ByPosition._index_rows(label)

    assert next(rows)['NAME'] == 'x'
    with pytest.raises(HostIndexError, match="NAME value mismatch: 'y', 'z'\nindex row "
                                             "1 does not match its supplement row"):
        next(rows)


@pytest.mark.parametrize('validate', [(), []], ids=['tuple', 'list'])
def test_an_empty_validation_compares_only_the_files(tmp_path: Path,
                                                     validate: tuple[str, ...] | list[str]
                                                     ) -> None:
    """The NAME columns differ, and only the files are compared."""

    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NAME': ['x', 'y']},
                    {'FILE_SPECIFICATION_NAME': _files('AB'), 'NAME': ['x', 'z']})

    assert [row['NAME'] for row in _ByPosition._index_rows(label, validate=validate)] \
        == ['x', 'y']


@pytest.mark.parametrize('host', [_ByPosition, _ByFile], ids=['by-position', 'by-file'])
def test_default_validation_accepts_a_label_and_its_data_file(tmp_path: Path,
                                                              host: type[Host]) -> None:
    """The index gives each data file and the supplement its label."""

    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': ['DATA/A.QUB', 'DATA/B.QUB'],
                               'NUM': ['1', '2']},
                    {'FILE_SPECIFICATION_NAME': ['DATA/A.LBL', 'DATA/B.LBL'],
                     'VAL': ['a', 'b']})

    assert _pairs(host._index_rows(label)) == [('1', 'a'), ('2', 'b')]


@pytest.mark.parametrize(('prefer_supplement', 'expected'),
                         [(False, ['i1', 'i2']), (True, ['s1', 's2'])])
def test_prefer_supplement_chooses_the_value_of_a_shared_column(
        tmp_path: Path, prefer_supplement: bool, expected: list[str]) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NAME': ['i1', 'i2']},
                    {'FILE_SPECIFICATION_NAME': _files('AB'), 'NAME': ['s1', 's2']})

    rows = _ByPosition._index_rows(label, validate=lambda indx, supp: True,
                                   prefer_supplement=prefer_supplement)

    assert [row['NAME'] for row in rows] == expected


##########################################################################################
# Pairing by file
##########################################################################################

def test_rows_in_another_order_pair_by_file(tmp_path: Path) -> None:
    """The supplement's paths differ from the index's in case, leading "/" and "./", the
    volume, given by another column, and the extension."""

    label = _tables(tmp_path, {'FILE_NAME': ['/VOL/DATA/A.QUB', '/VOL/DATA/B.QUB',
                                             '/VOL/DATA/C.QUB'],
                               'NUM': ['1', '2', '3']},
                    {'VOLUME_ID': ['vol', 'vol', 'vol'],
                     'FILE_SPECIFICATION_NAME': ['./data/c.lbl', 'data/a.lbl',
                                                 'data/b.lbl'],
                     'VAL': ['c', 'a', 'b']})

    rows = _ByFile._index_rows(label, file_key='FILE_NAME',
                               supplement_key=('VOLUME_ID', 'FILE_SPECIFICATION_NAME'),
                               validate=lambda indx, supp: True)

    assert _pairs(rows) == [('1', 'a'), ('2', 'b'), ('3', 'c')]


def test_pairing_by_file_holds_only_the_rows_read_ahead() -> None:
    """A supplement whose rows are 3 places out of order holds at most 3 rows."""

    k = 3
    index = [{'F': f'{i}.LBL', 'NUM': i} for i in range(12)]
    order = [block + k - j for block in range(0, 12, k + 1) for j in range(k + 1)]
    supplement = [{'F': f'{i}.LBL', 'VAL': i} for i in order]
    read = []

    def supplement_rows() -> Iterator[dict[str, Any]]:
        for row in supplement:
            read.append(row)
            yield row

    held = []
    rows = _index_rows.pair_by_file(iter(index), supplement_rows(), file_key='F',
                                    supplement_key='F', validate=None,
                                    prefer_supplement=False, files='index, supplement')
    for (i, row) in enumerate(rows):
        assert row['VAL'] == row['NUM']
        held.append(len(read) - (i + 1))

    assert max(held) == k


def test_pairing_by_file_raises_for_an_index_row_without_a_partner(
        tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NUM': ['1', '2']},
                    {'FILE_SPECIFICATION_NAME': _files('CA'), 'VAL': ['c', 'a']})
    rows = _ByFile._index_rows(label)

    assert next(rows)['VAL'] == 'a'
    with pytest.raises(HostIndexError, match="the supplement does not list the file of "
                                             "index row 1, 'B.LBL'"):
        next(rows)


def test_pairing_by_file_reports_an_index_row_that_repeats_a_file_as_unpaired(
        tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('ABA'),
                               'NUM': ['1', '2', '3']},
                    {'FILE_SPECIFICATION_NAME': _files('BAC'), 'VAL': ['b', 'a', 'c']})
    rows = _ByFile._index_rows(label)

    assert _pairs(next(rows) for _ in range(2)) == [('1', 'a'), ('2', 'b')]
    with pytest.raises(HostIndexError, match="index row 2, 'A.LBL', or lists it fewer "
                                             "times than the index"):
        next(rows)


def test_pairing_by_file_raises_at_a_repeated_supplement_row_while_the_first_is_held(
        tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('CA'),
                               'NUM': ['1', '2']},
                    {'FILE_SPECIFICATION_NAME': _files('AAC'), 'VAL': ['a', 'x', 'c']})
    rows = _ByFile._index_rows(label)

    with pytest.raises(HostIndexError, match="the supplement gives 'A.LBL' a second "
                                             "time, at row 1"):
        next(rows)


def test_pairing_by_file_raises_after_the_index_for_unpaired_supplement_rows(
        tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('AB'),
                               'NUM': ['1', '2']},
                    {'FILE_SPECIFICATION_NAME': _files('CBA'), 'VAL': ['c', 'b', 'a']})
    rows = _ByFile._index_rows(label)

    assert _pairs(next(rows) for _ in range(2)) == [('1', 'a'), ('2', 'b')]
    with pytest.raises(HostIndexError, match=r"the supplement lists files that the "
                                             r"index does not, or lists them more "
                                             r"times, such as \['C.LBL'\]"):
        next(rows)


def test_pairing_by_file_needs_a_file_key(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'NUM': ['1']}, {'NUM': ['1'], 'VAL': ['a']})

    with pytest.raises(HostIndexError, match='are paired by file, but the tables lack '
                                             'FILE_SPECIFICATION_NAME'):
        _ByFile._index_rows(label)


##########################################################################################
# What the call reads, and what the iteration reads
##########################################################################################

def test_the_call_reads_no_table_data(tmp_path: Path) -> None:
    """With both table files missing, the call succeeds; the iteration fails."""

    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('A'), 'NUM': ['1']},
                    {'FILE_SPECIFICATION_NAME': _files('A'), 'VAL': ['a']})
    for table in tmp_path.glob('*.tab'):
        table.unlink()

    rows = _ByPosition._index_rows(label, filter={'VAL': 'a'})

    with pytest.raises(FileNotFoundError, match='vol_index.tab'):
        next(rows)


def test_the_chunks_before_a_malformed_row_are_yielded(tmp_path: Path) -> None:
    """The first row of the second chunk is malformed."""

    names = [f'R{i:04d}' for i in range(_index_rows.CHUNK_ROWS + 1)]
    label = write_table(tmp_path, 'vol_index', {'NAME': names})
    table = tmp_path / 'vol_index.tab'
    table.write_bytes(table.read_bytes().replace(names[-1].encode(), b'XXXXXX'))
    rows = _ByPosition._index_rows(label, supplement=None)

    assert [next(rows)['NAME'] for _ in names[:-1]] == names[:-1]
    with pytest.raises(HostIndexError, match='invalid index table'):
        next(rows)


def test_rows_before_the_end_of_a_short_table_are_yielded(tmp_path: Path) -> None:
    """The label says five rows and the file holds two."""

    label = write_table(tmp_path, 'vol_index', {'NAME': ['A', 'B']}, label_rows=5)
    rows = _ByPosition._index_rows(label, supplement=None)

    assert [row['NAME'] for row in (next(rows), next(rows))] == ['A', 'B']
    with pytest.raises(HostIndexError, match='row count mismatch'):
        next(rows)


def test_a_host_identified_from_a_row_reads_only_the_first_row(tmp_path: Path) -> None:
    """The label says three rows and the file holds one, and the call still succeeds."""

    class _RowHost(Host):
        NAME = 'row-host'

        @staticmethod
        def _detect_in_row(row_dict: dict[str, Any]) -> bool:
            return bool(row_dict['NAME'] == 'A')

    label = write_table(tmp_path, 'vol_index', {'NAME': ['A']}, label_rows=3)

    rows = _RowHost._index_rows(label, supplement=None)

    assert next(rows)['NAME'] == 'A'


def test_an_index_of_another_host_raises_at_call_time(tmp_path: Path) -> None:
    label = write_table(tmp_path, 'vol_index', {'NAME': ['A']})

    class _NoHost(Host):
        NAME = 'no-host'

    with pytest.raises(WrongHostError, match='not a no-host index'):
        _NoHost._index_rows(label, supplement=None)


##########################################################################################
# Checks at call time
##########################################################################################

def test_a_filter_column_neither_table_has_raises_at_call_time(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('A'), 'NUM': ['1']},
                    {'FILE_SPECIFICATION_NAME': _files('A'), 'VAL': ['a']})

    with pytest.raises(OopsValueError, match="'NOPE'"):
        _ByPosition._index_rows(label, filter={'NOPE': 1})


def test_a_validation_column_a_table_lacks_raises_at_call_time(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('A'), 'NUM': ['1']},
                    {'FILE_SPECIFICATION_NAME': _files('A'), 'VAL': ['a']})

    with pytest.raises(HostIndexError, match='supplement lacks the column NUM'):
        _ByPosition._index_rows(label, validate='NUM')


def test_a_key_column_a_table_lacks_raises_at_call_time(tmp_path: Path) -> None:
    label = _tables(tmp_path, {'FILE_SPECIFICATION_NAME': _files('A'), 'NUM': ['1']},
                    {'FILE_SPECIFICATION_NAME': _files('A'), 'VAL': ['a']})

    with pytest.raises(HostIndexError, match='index lacks the column FILE_NAME'):
        _ByFile._index_rows(label, file_key='FILE_NAME')


##########################################################################################
# Filtering
##########################################################################################

ROWS: list[dict[str, Any]] = [{'NAME': 'A', 'NUM': 1}, {'NAME': 'B', 'NUM': 2},
                              {'NAME': 'C'}]


def test_given_rows_are_filtered_as_they_are_reached() -> None:
    """A row that lacks a column, as when its value is missing, fails the filter."""

    seen = []

    def given() -> Iterator[dict[str, Any]]:
        for row in ROWS:
            seen.append(row['NAME'])
            yield row

    rows = _ByPosition._index_rows('vol_index.lbl', row_dicts=given(),
                                   filter={'NUM': (2, 9)})

    assert next(rows)['NAME'] == 'B'
    assert seen == ['A', 'B']


def test_an_unknown_column_of_given_rows_raises_at_call_time() -> None:
    with pytest.raises(OopsValueError, match="'NOPE'"):
        _ByPosition._index_rows('vol_index.lbl', row_dicts=ROWS, filter={'NOPE': 1})


def test_an_unknown_column_of_an_iterator_of_rows_raises_at_the_first_row() -> None:
    """The call returns, because the columns are not known until a row is read."""

    rows = _ByPosition._index_rows('vol_index.lbl', row_dicts=iter(ROWS),
                                   filter={'NOPE': 1})

    with pytest.raises(OopsValueError, match="'NOPE'"):
        next(rows)


def test_a_range_that_cannot_be_compared_raises_at_its_row() -> None:
    rows = _ByPosition._index_rows('vol_index.lbl', row_dicts=ROWS,
                                   filter={'NUM': ('a', 'z')})

    with pytest.raises(OopsTypeError, match="column 'NUM' value 1"):
        next(rows)


@pytest.mark.parametrize(('filter_', 'expected'),
                         [({'NUM': 2}, ['B']), ({'NAME': {'A', 'C'}}, ['A', 'C']),
                          ({'NAME': 'A', 'NUM': (2, 3)}, [])],
                         ids=['value', 'set', 'both-columns'])
def test_a_filter_keeps_the_rows_that_match(filter_: dict[str, Any],
                                            expected: list[str]) -> None:
    rows = _ByPosition._index_rows('vol_index.lbl', row_dicts=ROWS, filter=filter_)

    assert [row['NAME'] for row in rows] == expected

##########################################################################################
