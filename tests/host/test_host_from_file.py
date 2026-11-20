##########################################################################################
# tests/host/test_host_from_file.py: Host.from_file hands the file to the host it finds
##########################################################################################

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from vicar import VicarImage

from oops.host import Host, PdsHostError

# The fast parser cannot read a quoted value that starts on the line after its "=" and
# continues onto another line; the strict parser can
FAST_FAILS_LABEL = """PDS_VERSION_ID = PDS3
DESCRIPTION =
    "first line
    second line."
END
"""

# No parser can read an unterminated quoted value
UNPARSEABLE_LABEL = """PDS_VERSION_ID = PDS3
DESCRIPTION = "unterminated
END
"""


class _FileHost(Host):
    """A host that claims every file in a standard format and records each call to its
    from_file."""

    NAME = 'file_host'
    calls: list[tuple[Any, dict[str, Any]]] = []

    @staticmethod
    def _detect_in_pds3(label: object) -> bool:
        return True

    @staticmethod
    def _detect_in_vicar(vic: object) -> bool:
        return True

    @staticmethod
    def from_file(fileinfo: Any, **kwargs: Any) -> str:
        _FileHost.calls.append((fileinfo, kwargs))
        return 'observation'


class _NameHost(Host):
    """A host that recognizes a file by its name alone."""

    NAME = 'name_host'
    calls: list[Any] = []

    @staticmethod
    def _detect_in_file(filepath: object) -> bool:
        return str(filepath).endswith('mine.lbl')

    @staticmethod
    def from_file(fileinfo: Any, **kwargs: Any) -> str:
        _NameHost.calls.append(fileinfo)
        return 'by name'


@pytest.fixture
def hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register _FileHost alone, with an empty record of calls."""

    monkeypatch.setattr(Host, '_HOSTS', [_FileHost])
    monkeypatch.setattr(_FileHost, 'calls', [])
    monkeypatch.setattr(_NameHost, 'calls', [])


def _label(tmp_path: Path, text: str, name: str = 'image.lbl') -> Path:
    filepath = tmp_path / name
    filepath.write_text(text)
    return filepath


@pytest.mark.usefixtures('hosts')
def test_the_host_receives_the_path_not_a_parsed_label(tmp_path: Path) -> None:
    """The host reads the file itself, so it parses the label the way it parses it when
    it is called directly."""

    filepath = _label(tmp_path, FAST_FAILS_LABEL)

    Host.from_file(filepath)

    assert _FileHost.calls[0][0] is filepath


@pytest.mark.usefixtures('hosts')
def test_the_host_receives_the_callers_pds3_options(tmp_path: Path) -> None:
    """No default is added to the PDS3 options, so the host applies its own."""

    Host.from_file(_label(tmp_path, FAST_FAILS_LABEL))

    assert _FileHost.calls[0][1]['pds3_options'] is None


@pytest.mark.usefixtures('hosts')
def test_calibrations_default_to_none(tmp_path: Path) -> None:
    """Without a calibrations override, the host's own calibration applies."""

    Host.from_file(_label(tmp_path, FAST_FAILS_LABEL))

    assert _FileHost.calls[0][1]['calibrations'] is None


@pytest.mark.usefixtures('hosts')
def test_a_label_that_only_a_later_parser_reads_is_identified(tmp_path: Path) -> None:
    """A label that the fast parser cannot read is identified with the strict parser."""

    assert Host.from_file(_label(tmp_path, FAST_FAILS_LABEL)) == 'observation'


@pytest.mark.usefixtures('hosts')
def test_an_unparseable_label_that_no_host_recognizes_raises(tmp_path: Path) -> None:
    with pytest.raises(PdsHostError, match='unterminated|Expected|missing'):
        Host.from_file(_label(tmp_path, UNPARSEABLE_LABEL))


@pytest.mark.usefixtures('hosts')
def test_an_unparseable_label_goes_to_a_host_that_knows_its_name(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A label that no parser can read is offered to each host's _detect_in_file, as a
    file in no standard format is, and the host that recognizes it receives its path."""

    monkeypatch.setattr(Host, '_HOSTS', [_FileHost, _NameHost])
    filepath = _label(tmp_path, UNPARSEABLE_LABEL, name='mine.lbl')

    assert Host.from_file(filepath) == 'by name'


@pytest.mark.usefixtures('hosts')
def test_a_vicar_file_is_identified_without_reading_its_array(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filepath = tmp_path / 'image.vic'
    VicarImage.from_array(np.zeros((2, 2), dtype=np.uint8)).write_file(filepath)

    def no_array_read(*args: object, **kwargs: object) -> None:
        raise AssertionError('the array was read')

    monkeypatch.setattr(VicarImage, 'from_file', no_array_read)

    assert Host.from_file(filepath) == 'observation'

##########################################################################################
