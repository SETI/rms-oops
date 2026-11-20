##########################################################################################
# tests/host/test_host_from_index.py: Host.from_index returns the host's iterator
##########################################################################################

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oops         import OopsValueError
from oops.host    import Host, HostIndexError, PdsHostError

INDEX_LABEL = """PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 5
FILE_RECORDS = 3
^INDEX_TABLE = "index.tab"
OBJECT = INDEX_TABLE
  INTERCHANGE_FORMAT = ASCII
  ROWS = 3
  ROW_BYTES = 5
  COLUMNS = 2
  OBJECT = COLUMN
    NAME = NAME
    DATA_TYPE = CHARACTER
    START_BYTE = 1
    BYTES = 1
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = NUM
    DATA_TYPE = ASCII_INTEGER
    START_BYTE = 3
    BYTES = 1
  END_OBJECT = COLUMN
END_OBJECT = INDEX_TABLE
END
"""


class _IndexHost(Host):
    """A host that claims every index and builds one name per row, recording each."""

    NAME = 'index_host'
    built: list[str] = []

    @staticmethod
    def _detect_in_index(label: object) -> bool | None:
        return True

    @staticmethod
    def from_index(filepath: Any, *, supplement: Any = Host.SUPPLEMENT,
                   label: Any = None, row_dicts: list[dict[str, Any]] | None = None,
                   pds3_options: dict[str, Any] | None = None,
                   filter: dict[str, Any] | None = None,  # noqa: A002
                   **kwargs: Any) -> Iterator[str]:
        rows = _IndexHost._index_rows(filepath, supplement=supplement,
                                      require_supplement=False, label=label,
                                      row_dicts=row_dicts, validate='NAME',
                                      pds3_options=pds3_options, filter=filter)

        def build() -> Iterator[str]:
            for row_dict in rows:
                _IndexHost.built.append(row_dict['NAME'])
                yield row_dict['NAME']
        return build()


class _RowHost(_IndexHost):
    """A host that can be identified only from the first row of an index."""

    NAME = 'row_host'

    @staticmethod
    def _detect_in_index(label: object) -> None:
        return None

    @staticmethod
    def _detect_in_row(row_dict: dict[str, Any]) -> bool:
        return bool(row_dict['NAME'] == 'A')


@pytest.fixture
def index_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A three-row index whose only registered host is _IndexHost."""

    monkeypatch.setattr(Host, '_HOSTS', [_IndexHost])
    monkeypatch.setattr(_IndexHost, 'built', [])
    (tmp_path / 'index.tab').write_bytes(b'A,1\r\nB,2\r\nC,3\r\n')
    label = tmp_path / 'index.lbl'
    label.write_text(INDEX_LABEL)
    return label


def test_from_index_returns_the_hosts_iterator_unconsumed(index_label: Path) -> None:
    """The result is an iterator over the filtered rows, and no row is built before it is
    reached."""

    result = Host.from_index(index_label, filter={'NUM': (2, 3)})

    assert isinstance(result, Iterator)
    assert _IndexHost.built == []
    assert next(result) == 'B'
    assert _IndexHost.built == ['B']
    assert list(result) == ['C']


def test_from_index_leaves_the_supplement_to_the_host(index_label: Path) -> None:
    """The host reads, checks, and filters the rows merged with the supplement's. NUM,
    which the two tables write differently, fails only the default check, which the host
    does not use."""

    supplement = index_label.with_name('supplement.lbl')
    supplement.with_suffix('.tab').write_bytes(b'A,9,x\r\nB,8,y\r\nC,7,z\r\n')
    val = ('  OBJECT = COLUMN\n    NAME = VAL\n    DATA_TYPE = CHARACTER\n'
           '    START_BYTE = 5\n    BYTES = 1\n  END_OBJECT = COLUMN\n')
    supplement.write_text(INDEX_LABEL.replace('index.tab', 'supplement.tab')
                                     .replace('BYTES = 5', 'BYTES = 7')
                                     .replace('COLUMNS = 2', 'COLUMNS = 3')
                                     .replace('END_OBJECT = INDEX_TABLE',
                                              val + 'END_OBJECT = INDEX_TABLE'))

    result = Host.from_index(index_label, supplement=supplement, filter={'VAL': 'y'})

    assert list(result) == ['B']


def test_from_index_raises_an_argument_error_at_call_time(index_label: Path) -> None:
    """An invalid PDS3 option is refused by the call itself, before any iteration."""

    with pytest.raises(OopsValueError, match='method'):
        Host.from_index(index_label, pds3_options={'method': 'bogus'})


def test_a_host_identified_from_a_row_is_found_from_the_first_row_alone(
        index_label: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The label says three rows and the file holds one, and the host is still found."""

    monkeypatch.setattr(Host, '_HOSTS', [_RowHost])
    table = index_label.with_name('index.tab')
    table.write_bytes(b'A,1\r\n')

    result = Host.from_index(index_label)

    assert next(result) == 'A'


def test_a_host_that_recognizes_the_label_is_found_before_any_row_is_read(
        index_label: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first host can tell only from a row, and the table file is missing, so the
    second host is found from the label alone."""

    monkeypatch.setattr(Host, '_HOSTS', [_RowHost, _IndexHost])
    index_label.with_name('index.tab').unlink()

    result = Host.from_index(index_label)

    with pytest.raises(FileNotFoundError, match='index.tab'):
        next(result)


def test_rows_given_by_the_caller_keep_the_row_read_to_find_the_host(
        index_label: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Host, '_HOSTS', [_RowHost])
    given = iter([{'NAME': 'A'}, {'NAME': 'B'}])

    assert list(Host.from_index(index_label, row_dicts=given)) == ['A', 'B']

##########################################################################################
# A volume's own index, and a label that cannot be parsed
##########################################################################################

class _MetadataHost(Host):
    """A host that reads only metadata indexes and recognizes a volume's own index from
    its label."""

    NAME = 'metadata_host'
    METADATA_INDEX_EXAMPLE = 'VOL_0001_index.lbl'

    @staticmethod
    def _detect_in_index(label: object) -> bool:
        return True

    @staticmethod
    def from_index(filepath: Any, *, label: Any = None,
                   **kwargs: Any) -> Iterator[dict[str, Any]]:
        return _MetadataHost._index_rows(filepath, supplement=None, label=label)


def test_a_volume_index_is_refused_without_reading_a_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first host could tell only from a row, and the table file is missing, so
    reading a row would raise FileNotFoundError."""

    monkeypatch.setattr(Host, '_HOSTS', [_RowHost, _MetadataHost])
    directory = tmp_path / 'VOL_0001' / 'INDEX'
    directory.mkdir(parents=True)
    label = directory / 'INDEX.LBL'
    label.write_text(INDEX_LABEL)

    with pytest.raises(HostIndexError, match='metadata_host reads the Ring-Moon Systems '
                                             'Node metadata index'):
        Host.from_index(label)


def test_a_label_that_cannot_be_parsed_and_no_host_claims_raises(
        index_label: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Host, '_HOSTS', [_IndexHost, _MetadataHost])
    index_label.write_text('PDS_VERSION_ID = PDS3\nRECORD_TYPE\nEND\n')

    with pytest.raises(PdsHostError, match=r'cannot parse the PDS3 label .*index\.lbl'):
        Host.from_index(index_label)

##########################################################################################
