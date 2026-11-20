##########################################################################################
# tests/host/test_host_label_correction.py: a host's correction of label text
##########################################################################################

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from oops.host import Host, PdsHostError
from tests.host._tables import write_table

# A line of plain text outside any comment, which no PDS3 parser can read, as in the label
# of a Cassini VIMS volume's own index
STRAY_LINE = 'For full definitions of index fields, please refer to the handbook.'


def _with_stray_line(label: Path) -> Path:
    """Insert STRAY_LINE after the first line of a label; return the label."""

    lines = label.read_text().split('\n')
    label.write_text('\n'.join([lines[0], STRAY_LINE, *lines[1:]]))
    return label


class _Fixing(Host):
    """A host whose correction removes STRAY_LINE, recording each call."""

    NAME = 'fixing'
    calls: list[tuple[str, Any]] = []
    built: list[Any] = []

    @classmethod
    def _correct_label_text(cls, text: str, filepath: Any) -> str:
        _Fixing.calls.append((text, filepath))
        return text.replace(STRAY_LINE + '\n', '')

    @staticmethod
    def _detect_in_pds3(label: Any) -> bool:
        return bool(label.get('INSTRUMENT_ID') == 'FIX')

    @staticmethod
    def _detect_in_index(label: Any) -> bool:
        return True

    @staticmethod
    def from_file(fileinfo: Any, **kwargs: Any) -> str:
        _Fixing.built.append(fileinfo)
        return 'fixed'

    @staticmethod
    def from_index(filepath: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        return _Fixing._index_rows(filepath, supplement=None)


class _Useless(_Fixing):
    """A host whose correction changes the text without making it readable."""

    NAME = 'useless'

    @classmethod
    def _correct_label_text(cls, text: str, filepath: Any) -> str:
        return text.replace('PDS_VERSION_ID', 'PDS_VERSION_ID ')


class _Plain(Host):
    """A host that keeps the default correction and claims nothing."""

    NAME = 'plain'

    @staticmethod
    def _detect_in_pds3(label: Any) -> bool:
        return False

    @staticmethod
    def _detect_in_index(label: Any) -> bool:
        return False


@pytest.fixture(autouse=True)
def _records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_Fixing, 'calls', [])
    monkeypatch.setattr(_Fixing, 'built', [])


def _product_label(directory: Path) -> Path:
    label = directory / 'image.lbl'
    label.write_text('PDS_VERSION_ID = PDS3\nINSTRUMENT_ID = FIX\n^IMAGE = "image.img"\n'
                     'END\n')
    return _with_stray_line(label)


def _index(directory: Path) -> Path:
    return write_table(directory, 'vol_index', {'NAME': ['A', 'B']})


def test_the_default_correction_leaves_the_text_unchanged() -> None:
    assert Host._correct_label_text('PDS_VERSION_ID = PDS3\nEND\n', 'x.lbl') == \
        'PDS_VERSION_ID = PDS3\nEND\n'


def test_the_correction_receives_the_text_and_the_path(tmp_path: Path) -> None:
    label = _product_label(tmp_path)
    _Fixing._read_fileinfo(label, formats='P', pds3_options={'method': 'fast'})
    assert _Fixing.calls == [(label.read_text(), FCPath(label))]


def test_a_product_label_is_parsed_after_the_correction(tmp_path: Path) -> None:
    parsed = _Fixing._read_fileinfo(_product_label(tmp_path), formats='P',
                                    pds3_options={'method': 'strict'})
    assert parsed['INSTRUMENT_ID'] == 'FIX'


def test_an_attached_label_is_parsed_after_the_correction(tmp_path: Path) -> None:
    """The label is followed by binary data, which is left out of the text."""

    filepath = tmp_path / 'image.img'
    filepath.write_bytes(f'PDS_VERSION_ID = PDS3\n{STRAY_LINE}\nRECORD_TYPE = UNDEFINED\n'
                         f'INSTRUMENT_ID = FIX\nEND\n'.encode() + bytes(range(256)))
    parsed = _Fixing._read_fileinfo(filepath, formats='P',
                                    pds3_options={'method': 'fast'})
    assert parsed['INSTRUMENT_ID'] == 'FIX'


def test_a_corrected_product_label_keeps_its_path(tmp_path: Path) -> None:
    label = _product_label(tmp_path)
    parsed = _Fixing._read_fileinfo(label, formats='P', pds3_options={'method': 'fast'})
    assert parsed.filepath == FCPath(label)


def test_an_index_label_is_parsed_after_the_correction(tmp_path: Path) -> None:
    rows = _Fixing._index_rows(_with_stray_line(_index(tmp_path)), supplement=None)
    assert [row['NAME'] for row in rows] == ['A', 'B']


def test_a_supplement_label_is_parsed_after_the_correction(tmp_path: Path) -> None:
    index = write_table(tmp_path, 'vol_index', {'FILE_SPECIFICATION_NAME': ['A.LBL'],
                                                'NUM': ['1']})
    supplement = write_table(tmp_path, 'vol_supplemental_index',
                             {'FILE_SPECIFICATION_NAME': ['A.LBL'], 'VAL': ['a']})
    _with_stray_line(supplement)

    assert [row['VAL'] for row in _Fixing._index_rows(index)] == ['a']


def test_the_dispatcher_gives_a_label_to_the_host_whose_correction_reads_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Host, '_HOSTS', [_Plain, _Useless, _Fixing])
    assert Host.from_file(_product_label(tmp_path)) == 'fixed'


def test_the_dispatcher_gives_an_index_to_the_host_whose_correction_reads_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Host, '_HOSTS', [_Plain, _Useless, _Fixing])
    rows = Host.from_index(_with_stray_line(_index(tmp_path)))
    assert [row['NAME'] for row in rows] == ['A', 'B']


def test_a_correction_that_does_not_help_raises_the_parse_error(tmp_path: Path) -> None:
    with pytest.raises(PdsHostError, match=r'cannot parse the PDS3 label .*image\.lbl'):
        _Useless._read_fileinfo(_product_label(tmp_path), formats='P',
                                pds3_options={'method': 'fast'})


def test_the_dispatcher_raises_the_parse_error_if_no_correction_helps(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Host, '_HOSTS', [_Plain, _Useless])
    with pytest.raises(PdsHostError, match=r'cannot parse the PDS3 label .*vol_index\.lbl'):
        Host.from_index(_with_stray_line(_index(tmp_path)))

##########################################################################################
