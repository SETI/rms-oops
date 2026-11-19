##########################################################################################
# tests/host/test_host_index_rows.py: pairing the rows of an index and its supplement
##########################################################################################

from pathlib import Path

import pytest

from oops.host import Host, HostIndexError


def _write_table(directory: Path, stem: str, columns: dict[str, list[str]]) -> Path:
    """Write an ASCII table of character columns and its detached label; return the label.

    Parameters:
        directory: The directory to write into.
        stem: The file name without its extension.
        columns: The values of each column, by column name.
    """

    names = list(columns)
    widths = [max(len(value) for value in columns[name]) for name in names]
    nrows = len(columns[names[0]])
    row_bytes = sum(widths) + len(widths) - 1 + 2
    lines = [','.join(columns[name][i].ljust(width) for name, width in zip(names, widths))
             for i in range(nrows)]
    (directory / f'{stem}.tab').write_bytes(''.join(line + '\r\n'
                                                    for line in lines).encode())

    start = 1
    objects = []
    for name, width in zip(names, widths):
        objects += ['  OBJECT = COLUMN', f'    NAME = {name}',
                    '    DATA_TYPE = CHARACTER', f'    START_BYTE = {start}',
                    f'    BYTES = {width}', '  END_OBJECT = COLUMN']
        start += width + 1
    label = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
             f'RECORD_BYTES = {row_bytes}', f'FILE_RECORDS = {nrows}',
             f'^INDEX_TABLE = "{stem}.tab"', 'OBJECT = INDEX_TABLE',
             '  INTERCHANGE_FORMAT = ASCII', f'  ROWS = {nrows}',
             f'  ROW_BYTES = {row_bytes}', f'  COLUMNS = {len(names)}',
             *objects, 'END_OBJECT = INDEX_TABLE', 'END', '']
    filepath = directory / f'{stem}.lbl'
    filepath.write_text('\n'.join(label))
    return filepath


def _read(directory: Path, index: dict[str, list[str]], supplement: dict[str, list[str]],
          **kwargs: object) -> list[dict[str, object]]:
    """Write an index and its supplement and return the rows that pair them."""

    label = _write_table(directory, 'vol_index', index)
    _write_table(directory, 'vol_supplemental_index', supplement)
    return Host._read_index_rows(label, supplement=Host.SUPPLEMENT, **kwargs)


def test_rows_in_the_same_order_pair_by_position(tmp_path: Path) -> None:
    rows = _read(tmp_path, {'FILE_SPECIFICATION_NAME': ['A.LBL', 'B.LBL'],
                            'NUM': ['1', '2']},
                 {'FILE_SPECIFICATION_NAME': ['A.LBL', 'B.LBL'], 'VAL': ['a', 'b']})
    assert [(row['NUM'], row['VAL']) for row in rows] == [('1', 'a'), ('2', 'b')]


def test_rows_in_another_order_pair_by_file(tmp_path: Path) -> None:
    """The supplement's paths differ from the index's in case, leading "/" and "./", the
    volume, given by another column, and the extension."""

    rows = _read(tmp_path, {'FILE_NAME': ['/VOL/DATA/A.QUB', '/VOL/DATA/B.QUB',
                                          '/VOL/DATA/C.QUB'],
                            'NUM': ['1', '2', '3']},
                 {'VOLUME_ID': ['vol', 'vol', 'vol'],
                  'FILE_SPECIFICATION_NAME': ['./data/c.lbl', 'data/a.lbl', 'data/b.lbl'],
                  'VAL': ['c', 'a', 'b']},
                 file_key='FILE_NAME',
                 supplement_key=('VOLUME_ID', 'FILE_SPECIFICATION_NAME'),
                 validate=lambda indx, supp: True)
    assert [(row['NUM'], row['VAL']) for row in rows] == [('1', 'a'), ('2', 'b'),
                                                         ('3', 'c')]


@pytest.mark.parametrize('order', [[0, 1], [1, 0]], ids=['same-order', 'reversed'])
def test_default_validation_accepts_a_label_and_its_data_file(tmp_path: Path,
                                                              order: list[int]) -> None:
    """The index names each data file and the supplement its label."""

    labels = ['DATA/A.LBL', 'DATA/B.LBL']
    rows = _read(tmp_path, {'FILE_SPECIFICATION_NAME': ['DATA/A.QUB', 'DATA/B.QUB'],
                            'NUM': ['1', '2']},
                 {'FILE_SPECIFICATION_NAME': [labels[i] for i in order],
                  'VAL': [['a', 'b'][i] for i in order]})
    assert [(row['NUM'], row['VAL']) for row in rows] == [('1', 'a'), ('2', 'b')]


def test_an_index_row_without_a_supplement_row_raises(tmp_path: Path) -> None:
    with pytest.raises(HostIndexError,
                       match=r"no supplement row for an index row \(1\): \['B.LBL'\]"):
        _read(tmp_path, {'FILE_SPECIFICATION_NAME': ['A.LBL', 'B.LBL'],
                         'NUM': ['1', '2']},
              {'FILE_SPECIFICATION_NAME': ['C.LBL', 'A.LBL'], 'VAL': ['c', 'a']})


def test_a_file_listed_twice_raises(tmp_path: Path) -> None:
    with pytest.raises(HostIndexError,
                       match=r"supplement lists a file more than once \(1\): "
                             r"\['A.LBL'\]"):
        _read(tmp_path, {'FILE_SPECIFICATION_NAME': ['B.LBL', 'A.LBL'],
                         'NUM': ['1', '2']},
              {'FILE_SPECIFICATION_NAME': ['A.LBL', 'A.LBL'], 'VAL': ['a', 'b']})


@pytest.mark.parametrize(('prefer_supplement', 'expected'),
                         [(False, ['i1', 'i2']), (True, ['s1', 's2'])])
def test_prefer_supplement_chooses_the_value_of_a_shared_column(
        tmp_path: Path, prefer_supplement: bool, expected: list[str]) -> None:
    rows = _read(tmp_path, {'FILE_SPECIFICATION_NAME': ['A.LBL', 'B.LBL'],
                            'NAME': ['i1', 'i2']},
                 {'FILE_SPECIFICATION_NAME': ['B.LBL', 'A.LBL'], 'NAME': ['s2', 's1']},
                 validate='FILE_SPECIFICATION_NAME', prefer_supplement=prefer_supplement)
    assert [row['NAME'] for row in rows] == expected

##########################################################################################
