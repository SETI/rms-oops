##########################################################################################
# tests/host/test_host_from_index.py: Host.from_index identifies the host and hands over
##########################################################################################

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oops.host import Host, HostError, UnknownHostError
from tests.host._tables import write_table


class _Recorder(Host):
    """A host whose from_index records its arguments and yields the NUM of each row."""

    NAME = 'recorder'
    calls: list[dict[str, Any]] = []

    @staticmethod
    def _detect_in_index(label: object) -> bool | None:
        return True

    @classmethod
    def from_index(cls, filepath: Any, **kwargs: Any) -> Iterator[Any]:
        cls.calls.append({'filepath': filepath, **kwargs})
        rows = cls._index_rows(filepath, supplement=kwargs['supplement'],
                               require_supplement=False, label=kwargs['label'],
                               row_dicts=kwargs['row_dicts'],
                               pds3_options=kwargs['pds3_options'],
                               filter=kwargs['filter'])
        return (str(row['NUM']).strip() for row in rows)


class _Other(_Recorder):
    """A second host that claims every index."""

    NAME = 'other'


class _Declines(_Recorder):
    """A host that recognizes no index."""

    NAME = 'declines'

    @staticmethod
    def _detect_in_index(label: object) -> bool | None:
        return False


class _RowOne(_Recorder):
    """A host that recognizes an index only from a first row whose NUM is 1."""

    NAME = 'row_one'

    @staticmethod
    def _detect_in_index(label: object) -> bool | None:
        return None

    @staticmethod
    def _detect_in_row(row_dict: dict[str, Any]) -> bool:
        return bool(str(row_dict['NUM']).strip() == '1')


class _RowAny(_RowOne):
    """A second host that recognizes an index only from its first row."""

    NAME = 'row_any'

    @staticmethod
    def _detect_in_row(row_dict: dict[str, Any]) -> bool:
        return True


@pytest.fixture
def hosts(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A function that sets the registered hosts and clears the recorded calls."""

    def register(*registered: type[Host]) -> None:
        monkeypatch.setattr(Host, '_HOSTS', list(registered))
        monkeypatch.setattr(_Recorder, 'calls', [])

    return register


@pytest.fixture
def index(tmp_path: Path) -> Path:
    """A three-row index whose NUM column is 1, 2 and 3."""

    return write_table(tmp_path, 'vol_index', {'NUM': ['1', '2', '3']},
                       integers=('NUM',))


def test_from_index_returns_the_hosts_iterator(hosts: Any, index: Path) -> None:
    hosts(_Declines, _Recorder)
    result = Host.from_index(index, filter={'NUM': (2, 3)})

    assert isinstance(result, Iterator)
    assert list(result) == ['2', '3']


def test_every_argument_is_handed_to_the_host_as_given(hosts: Any, index: Path) -> None:
    hosts(_Recorder)
    list(Host.from_index(index, supplement=None, select=1, target='SATURN',
                         timeshift=(1., 'T'), extra='x'))
    call = _Recorder.calls[0]

    assert call.pop('filepath').as_posix() == index.as_posix()
    assert call == {'supplement': None, 'label': None,
                    'row_dicts': None, 'pds3_options': None, 'filter': None,
                    'select': 1, 'target': 'SATURN', 'lightsource': None, 'path': None,
                    'frame': None, 'cadence': None, 'fov': None, 'calibrations': None,
                    'timeshift': (1., 'T'), 'navigation': None, 'tracker': None,
                    'parallel': None, 'extra': 'x'}


def test_a_label_the_caller_gave_is_handed_on(hosts: Any, index: Path) -> None:
    hosts(_Recorder)
    parsed = Host._parse_pds3(index, {})
    list(Host.from_index(index, label=parsed))

    assert _Recorder.calls[0]['label'] is parsed


def test_an_index_that_two_hosts_claim_is_refused(hosts: Any, index: Path) -> None:
    hosts(_Recorder, _Declines, _Other)

    with pytest.raises(HostError, match='index .*vol_index.lbl is claimed by more than '
                                        'one host: recorder, other'):
        Host.from_index(index)


def test_the_first_row_identifies_a_host_the_label_cannot(hosts: Any,
                                                          index: Path) -> None:
    hosts(_Declines, _RowOne)

    assert list(Host.from_index(index)) == ['1', '2', '3']


def test_a_first_row_that_two_hosts_claim_is_refused(hosts: Any, index: Path) -> None:
    hosts(_RowOne, _RowAny)

    with pytest.raises(HostError, match='first row of the index .*vol_index.lbl is '
                                        'claimed by more than one host: row_one, '
                                        'row_any'):
        Host.from_index(index)


def test_an_index_that_no_host_claims_is_unknown(hosts: Any, index: Path) -> None:
    hosts(_Declines)

    with pytest.raises(UnknownHostError, match='unknown host in index .*vol_index.lbl'):
        Host.from_index(index)


def test_a_first_row_that_no_host_claims_is_unknown(hosts: Any, tmp_path: Path) -> None:
    hosts(_RowOne)
    index = write_table(tmp_path, 'vol_index', {'NUM': ['2']}, integers=('NUM',))

    with pytest.raises(UnknownHostError, match='unknown host in index'):
        Host.from_index(index)


def test_an_empty_index_that_only_a_row_could_identify_is_unknown(
        hosts: Any, tmp_path: Path) -> None:
    hosts(_RowAny)
    index = write_table(tmp_path, 'vol_index', {'NUM': ['1']}, label_rows=0)
    (tmp_path / 'vol_index.tab').write_bytes(b'')

    with pytest.raises(UnknownHostError, match='unknown host in index'):
        Host.from_index(index)


@pytest.mark.parametrize('make', [list, iter], ids=['list', 'iterator'])
def test_given_rows_still_hold_the_row_read_to_identify_the_host(
        hosts: Any, index: Path, make: Any) -> None:
    hosts(_RowOne)
    rows = make([{'NUM': 1}, {'NUM': 2}])

    assert list(Host.from_index(index, row_dicts=rows)) == ['1', '2']


def test_no_given_rows_leave_the_host_unknown(hosts: Any, index: Path) -> None:
    hosts(_RowAny)

    with pytest.raises(UnknownHostError, match='unknown host in index'):
        Host.from_index(index, row_dicts=iter([]))

##########################################################################################
