##########################################################################################
# tests/host/test_host_from_file.py: Host.from_file, its claimants, and label corrections
##########################################################################################

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits
from filecache  import FCPath
from vicar      import VicarImage, VicarLabel

from oops      import OopsValueError
from oops.host import Host, HostError, PdsHostError, host_
from tests.host._tables import write_table

LABEL = 'PDS_VERSION_ID = PDS3\nINSTRUMENT_ID = CAM_A\n^IMAGE = "image.img"\nEND\n'
PROSE = 'a line of prose\n'
BROKEN_LABEL = LABEL.replace('END\n', PROSE + 'END\n')
BAD_DAY = '2003-02-30T00:00:00'
BAD_DAY_LABEL = LABEL.replace('END\n', f'START_TIME = {BAD_DAY}\nEND\n')

RECEIVED: list[tuple[Any, dict[str, Any]]] = []


class _CamA(Host):
    """A host that claims the label of CAM_A and records what its from_file receives."""

    NAME = 'cam_a'

    @staticmethod
    def _detect_in_pds3(label: Any) -> bool:
        return bool(label.get('INSTRUMENT_ID') == 'CAM_A')

    @classmethod
    def from_file(cls, fileinfo: Any, **kwargs: Any) -> Any:
        RECEIVED.append((fileinfo, kwargs))
        return cls._read_fileinfo(fileinfo, formats='PVFO',
                                  pds3_options=kwargs['pds3_options'])


class _Corrects(_CamA):
    """A host of CAM_A whose correction removes the line of prose."""

    NAME = 'corrects'

    @classmethod
    def _correct_label_text(cls, text: str, filepath: FCPath) -> str:
        return text.replace(PROSE, '')


class _CorrectsDay(_CamA):
    """A host of CAM_A whose correction replaces an impossible date."""

    NAME = 'corrects_day'

    @classmethod
    def _correct_label_text(cls, text: str, filepath: FCPath) -> str:
        return text.replace(BAD_DAY, '2003-03-02T00:00:00')


class _AlsoCorrects(_Corrects):
    """A second host of CAM_A whose correction removes the line of prose."""

    NAME = 'also_corrects'


class _CorrectsOther(_Corrects):
    """A host whose correction removes the prose but which claims only CAM_B."""

    NAME = 'corrects_other'

    @staticmethod
    def _detect_in_pds3(label: Any) -> bool:
        return bool(label.get('INSTRUMENT_ID') == 'CAM_B')


class _CorrectsNothing(_CamA):
    """A host whose correction changes nothing."""

    NAME = 'corrects_nothing'

    @classmethod
    def _correct_label_text(cls, text: str, filepath: FCPath) -> str:
        return text


class _ClaimsAll(Host):
    """A host that claims every file of every format."""

    NAME = 'claims_all'

    @staticmethod
    def _detect_in_pds3(label: Any) -> bool:
        return True

    @staticmethod
    def _detect_in_vicar(label: Any) -> bool:
        return True

    @staticmethod
    def _detect_in_fits(hdulist: Any) -> bool:
        return True

    @staticmethod
    def _detect_in_file(filepath: Any) -> bool:
        return True

    @classmethod
    def from_file(cls, fileinfo: Any, **kwargs: Any) -> Any:
        RECEIVED.append((fileinfo, kwargs))
        return None


class _AlsoClaimsAll(_ClaimsAll):
    """A second host that claims every file of every format."""

    NAME = 'also_claims_all'


@pytest.fixture
def hosts(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """A function that sets the registered hosts and clears what they received."""

    def register(*registered: type[Host]) -> None:
        monkeypatch.setattr(Host, '_HOSTS', list(registered))
        RECEIVED.clear()

    return register


def _write(directory: Path, name: str, text: str) -> Path:
    filepath = directory / name
    filepath.write_text(text)
    return filepath


def _write_label(directory: Path) -> Path:
    return _write(directory, 'image.lbl', LABEL)


def _write_fits(directory: Path) -> Path:
    filepath = directory / 'image.fits'
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint8)).writeto(filepath)
    return filepath


def _write_vicar(directory: Path) -> Path:
    filepath = directory / 'image.vic'
    VicarImage.from_array(np.zeros((2, 2), dtype=np.uint8)).write_file(filepath)
    return filepath


def _write_other(directory: Path) -> Path:
    return _write(directory, 'image.dat', '\x00' * 16)

##########################################################################################
# What the host receives
##########################################################################################

def test_a_detached_label_is_passed_to_the_host_as_its_path(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA)
    filepath = _write_label(tmp_path)
    Host.from_file(filepath)

    assert RECEIVED[0][0] is filepath


def test_an_attached_label_is_passed_to_the_host_as_its_path(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA)
    filepath = _write(tmp_path, 'image.img', LABEL)
    Host.from_file(filepath)

    assert RECEIVED[0][0] is filepath


def test_the_host_receives_the_pds3_options_as_the_caller_gave_them(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA)
    Host.from_file(_write_label(tmp_path))

    assert RECEIVED[0][1]['pds3_options'] is None


def test_every_other_argument_is_handed_to_the_host_as_given(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA)
    Host.from_file(_write_label(tmp_path), select=0, target='SATURN', extra='x')
    kwargs = RECEIVED[0][1]

    assert kwargs == {'pds3_options': None, 'select': 0, 'astrometry': False,
                      'target': 'SATURN', 'lightsource': None, 'path': None,
                      'frame': None, 'cadence': None, 'fov': None, 'calibrations': [],
                      'timeshift': None, 'navigation': None, 'tracker': None,
                      'parallel': None, 'extra': 'x'}


def test_a_vicar_file_is_passed_to_the_host_as_its_content(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_ClaimsAll)
    Host.from_file(_write_vicar(tmp_path), astrometry=True)

    assert isinstance(RECEIVED[0][0], VicarLabel)


def test_a_fits_file_is_passed_to_the_host_as_its_content(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_ClaimsAll)
    Host.from_file(_write_fits(tmp_path))

    assert isinstance(RECEIVED[0][0], fits.HDUList)


def test_a_file_of_another_format_is_passed_to_the_host_as_its_path(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_ClaimsAll)
    filepath = _write_other(tmp_path)
    Host.from_file(filepath)

    assert RECEIVED[0][0].as_posix() == filepath.as_posix()


def test_content_already_parsed_is_returned_as_it_is(tmp_path: Path) -> None:
    label = Host._parse_pds3(FCPath(_write_label(tmp_path)), {})

    assert _CamA._read_fileinfo(label, formats='P') is label


def test_the_default_correction_returns_the_text_unchanged() -> None:
    assert Host._correct_label_text(LABEL, FCPath('image.lbl')) is LABEL

##########################################################################################
# More than one claimant
##########################################################################################

@pytest.mark.parametrize(('writer', 'kind'),
                         [(_write_label, 'PDS3 label'), (_write_fits, 'FITS file'),
                          (_write_vicar, 'VICAR file'), (_write_other, 'file')],
                         ids=['pds3', 'fits', 'vicar', 'other'])
def test_a_file_that_two_hosts_claim_is_refused(
        tmp_path: Path, hosts: Callable[..., None], writer: Callable[[Path], Path],
        kind: str) -> None:
    hosts(_ClaimsAll, _AlsoClaimsAll)
    filepath = writer(tmp_path)

    with pytest.raises(HostError, match=f'{kind} .*{filepath.name} is claimed by more '
                                        f'than one host: claims_all, also_claims_all'):
        Host.from_file(filepath)

##########################################################################################
# Corrections of label text when a label cannot be parsed
##########################################################################################

def test_a_host_whose_correction_reads_the_label_claims_it(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA, _CorrectsOther, _Corrects)
    filepath = _write(tmp_path, 'image.lbl', BROKEN_LABEL)
    label = Host.from_file(filepath)

    assert label['INSTRUMENT_ID'] == 'CAM_A'


def test_a_corrected_label_keeps_the_path_of_its_file(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_Corrects)
    filepath = _write(tmp_path, 'image.lbl', BROKEN_LABEL)
    label = Host.from_file(filepath)

    assert label.filepath.as_posix() == filepath.as_posix()


def test_a_label_with_an_impossible_date_is_read_by_a_correction(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CorrectsDay)
    filepath = _write(tmp_path, 'image.lbl', BAD_DAY_LABEL)
    label = Host.from_file(filepath)

    assert label['START_TIME'].isoformat() == '2003-03-02T00:00:00'


def test_a_label_with_an_impossible_date_that_no_correction_reads_is_refused(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA, _CorrectsNothing)
    filepath = _write(tmp_path, 'image.lbl', BAD_DAY_LABEL)

    with pytest.raises(OopsValueError, match='day is out of range for month'):
        Host.from_file(filepath)


def test_a_label_that_two_corrections_read_is_refused(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_Corrects, _CamA, _AlsoCorrects)
    filepath = _write(tmp_path, 'image.lbl', BROKEN_LABEL)

    with pytest.raises(HostError, match='PDS3 label .*image.lbl is claimed by more than '
                                        'one host: corrects, also_corrects'):
        Host.from_file(filepath)


def test_a_label_that_no_correction_reads_raises_the_parse_error(
        tmp_path: Path, hosts: Callable[..., None]) -> None:
    hosts(_CamA, _CorrectsNothing, _CorrectsOther)
    filepath = _write(tmp_path, 'image.lbl', BROKEN_LABEL)

    with pytest.raises(PdsHostError, match='cannot parse the PDS3 label .*image.lbl: '
                                           "Expected 'END', found 'a'"):
        Host.from_file(filepath)


def test_a_host_parses_its_own_label_with_its_correction(tmp_path: Path) -> None:
    filepath = _write(tmp_path, 'image.lbl', BROKEN_LABEL)
    label = _Corrects._read_fileinfo(filepath, formats='P')

    assert label['INSTRUMENT_ID'] == 'CAM_A'


def test_a_label_that_the_hosts_own_correction_cannot_read_is_refused(
        tmp_path: Path) -> None:
    filepath = _write(tmp_path, 'image.lbl', BROKEN_LABEL)

    with pytest.raises(PdsHostError, match='cannot parse the PDS3 label .*image.lbl'):
        _CorrectsNothing._read_fileinfo(filepath, formats='P')


def test_an_index_label_is_parsed_as_it_is_written(tmp_path: Path) -> None:
    """The correction would break the label, so the index is read only if it is unused."""

    class _BreaksLabels(Host):
        NAME = 'breaks_labels'

        @staticmethod
        def _detect_in_index(label: object) -> bool:
            return True

        @classmethod
        def _correct_label_text(cls, text: str, filepath: FCPath) -> str:
            return 'not a label'

    label = write_table(tmp_path, 'vol_index', {'NUM': ['1']})

    assert [str(row['NUM']) for row in
            _BreaksLabels._index_rows(label, supplement=None)] == ['1']

##########################################################################################
# Reading the text of a label for a correction
##########################################################################################

def _upper(text: str, filepath: FCPath) -> str:
    return text.upper()


def test_the_text_of_an_attached_label_without_end_cannot_be_read(
        tmp_path: Path) -> None:
    filepath = _write(tmp_path, 'image.img', 'PDS_VERSION_ID = PDS3\n')

    with pytest.raises(PdsHostError, match='cannot parse the PDS3 label .*image.img: '
                                           'missing END statement'):
        Host._parse_pds3(FCPath(filepath), {}, correct=_upper)


def test_a_compound_label_is_read_as_its_whole_text(tmp_path: Path) -> None:
    filepath = FCPath(_write(tmp_path, 'image.lbl', LABEL + 'trailing text\n'))
    (text, _) = Host._pds3_source(filepath, {'method': 'compound'}, _upper)

    assert text.endswith('TRAILING TEXT\n')


def test_a_vax_label_is_read_with_the_vax_reader(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_, 'read_vax_binary_label', lambda filepath: 'vax text\n')
    (text, options) = Host._pds3_source(FCPath(tmp_path / 'image.lbl'), {'vax': True},
                                        _upper)

    assert (text, options) == ('VAX TEXT\n', {'fmt_dirs': [FCPath(tmp_path)]})


def test_a_corrected_label_finds_the_format_file_beside_it(tmp_path: Path) -> None:
    text = ('PDS_VERSION_ID = PDS3\nOBJECT = TABLE\n  ^STRUCTURE = "COLUMNS.FMT"\n'
            'END_OBJECT = TABLE\nEND\n')
    _write(tmp_path, 'COLUMNS.FMT', 'OBJECT = COLUMN\n  NAME = FIRST\n'
                                    'END_OBJECT = COLUMN\nEND\n')
    filepath = FCPath(_write(tmp_path, 'image.lbl', text))
    label = Host._parse_pds3(filepath, {'expand': True}, correct=_upper)

    assert label['TABLE']['FIRST']['NAME'] == 'FIRST'


def test_corrected_text_of_one_line_is_still_read_as_text(tmp_path: Path) -> None:
    filepath = FCPath(_write_label(tmp_path))
    (text, _) = Host._pds3_source(filepath, {}, lambda text, path: 'END')

    assert text == 'END\n'


def test_an_invalid_parser_option_is_a_value_error(tmp_path: Path) -> None:
    filepath = FCPath(_write_label(tmp_path))

    with pytest.raises(OopsValueError, match='invalid method'):
        Host._parse_pds3(filepath, {'method': 'bogus'})


##########################################################################################
