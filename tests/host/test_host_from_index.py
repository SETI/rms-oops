##########################################################################################
# tests/host/test_host_from_index.py: Host.from_index returns the host's iterator
##########################################################################################

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oops         import OopsValueError
from oops.host    import Host

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
    def _detect_in_index(label: object) -> bool:
        return True

    @staticmethod
    def from_index(filepath: Any, *, row_dicts: list[dict[str, Any]],
                   **kwargs: Any) -> Iterator[str]:
        def build() -> Iterator[str]:
            for row_dict in row_dicts:
                _IndexHost.built.append(row_dict['NAME'])
                yield row_dict['NAME']
        return build()


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


def test_from_index_raises_an_argument_error_at_call_time(index_label: Path) -> None:
    """An invalid PDS3 option is refused by the call itself, before any iteration."""

    with pytest.raises(OopsValueError, match='method'):
        Host.from_index(index_label, pds3_options={'method': 'bogus'})

##########################################################################################
