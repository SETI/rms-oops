##########################################################################################
# tests/host/test_host_identify.py: Host file identification and index reading
##########################################################################################

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from filecache  import FCPath
from vicar      import VicarImage

from oops.host import (AmbiguousHostError, Host, HostFormatError, HostIndexError,
                       PdsHostError, UnknownHostError, WrongHostError)

INDEX_LABEL = """PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 5
FILE_RECORDS = 2
^INDEX_TABLE = "{table}"
OBJECT = INDEX_TABLE
  INTERCHANGE_FORMAT = ASCII
  ROWS = 2
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

ATTACHED_LABEL = """PDS_VERSION_ID = PDS3
RECORD_TYPE = UNDEFINED
END
"""


class _HostA(Host):
    """An unregistered host that recognizes no file."""

    NAME = 'host_a'

    @staticmethod
    def _detect_in_pds3(label: object) -> bool:
        return False

    @staticmethod
    def _detect_in_fits(hdulist: object) -> bool:
        return False

    @staticmethod
    def _detect_in_vicar(vic: object) -> bool:
        return False

    @staticmethod
    def _detect_in_file(filepath: object) -> bool:
        return False


class _HostB(_HostA):
    """A second unregistered host that recognizes no file."""

    NAME = 'host_b'


class _ClaimsAll(Host):
    """An unregistered host that recognizes every file."""

    NAME = 'claims_all'

    @staticmethod
    def _detect_in_pds3(label: object) -> bool:
        return True

    @staticmethod
    def _detect_in_fits(hdulist: object) -> bool:
        return True

    @staticmethod
    def _detect_in_vicar(vic: object) -> bool:
        return True

    @staticmethod
    def _detect_in_file(filepath: object) -> bool:
        return True


class _ClaimsAllToo(_ClaimsAll):
    """A second unregistered host that recognizes every file."""

    NAME = 'claims_all_too'


def _write_index(directory: Path, stem: str) -> Path:
    """Write a two-row ASCII index table and its detached label; return the label."""

    (directory / f'{stem}.tab').write_bytes(b'A,1\r\nB,2\r\n')
    label = directory / f'{stem}.lbl'
    label.write_text(INDEX_LABEL.format(table=f'{stem}.tab'))
    return label


def _write_pds3_label(directory: Path) -> Path:
    label = directory / 'image.lbl'
    label.write_text(INDEX_LABEL.format(table='image.tab'))
    return label


def _write_attached_pds3(directory: Path) -> Path:
    filepath = directory / 'image.img'
    filepath.write_text(ATTACHED_LABEL)
    return filepath


def _write_fits(directory: Path) -> Path:
    filepath = directory / 'image.fits'
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint8)).writeto(filepath)
    return filepath


def _write_vicar(directory: Path) -> Path:
    filepath = directory / 'image.vic'
    VicarImage.from_array(np.zeros((2, 2), dtype=np.uint8)).write_file(filepath)
    return filepath


def _write_other(directory: Path) -> Path:
    filepath = directory / 'image.dat'
    filepath.write_bytes(b'\x00' * 16)
    return filepath


FILE_WRITERS = [
    pytest.param(_write_pds3_label, 'PDS3 label', id='detached-pds3'),
    pytest.param(_write_attached_pds3, 'PDS3 file', id='attached-pds3'),
    pytest.param(_write_fits, 'FITS file', id='fits'),
    pytest.param(_write_vicar, 'VICAR file', id='vicar'),
    pytest.param(_write_other, 'file', id='other'),
]

##########################################################################################
# A missing FITS file is an OSError, not a format error
##########################################################################################

def test_a_missing_fits_file_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='missing.fits'):
        Host._identify_and_parse(tmp_path / 'missing.fits', hosts=[_HostA])


def test_an_unreadable_fits_file_is_a_format_error(tmp_path: Path) -> None:
    filepath = tmp_path / 'garbage.fits'
    filepath.write_bytes(b'not a FITS file')
    with pytest.raises(HostFormatError, match='not a valid FITS file'):
        Host._identify_and_parse(filepath, hosts=[_HostA])


def test_a_fits_file_that_no_host_claims_is_closed(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[fits.HDUList] = []
    real_open = fits.open

    def recording_open(*args: object, **kwargs: object) -> fits.HDUList:
        hdulist = real_open(*args, **kwargs)
        opened.append(hdulist)
        return hdulist

    monkeypatch.setattr(fits, 'open', recording_open)
    with pytest.raises(WrongHostError):
        Host._identify_and_parse(_write_fits(tmp_path), hosts=[_HostA])

    assert opened[0]._file.closed

##########################################################################################
# A label that cannot be parsed is named in the error
##########################################################################################

def test_an_index_label_that_cannot_be_parsed_is_named(tmp_path: Path) -> None:
    label = tmp_path / 'vol_index.lbl'
    label.write_text('PDS_VERSION_ID = PDS3\nRECORD_TYPE\nEND\n')
    with pytest.raises(PdsHostError,
                       match=r'^cannot parse the PDS3 label .*vol_index\.lbl: '
                             r'missing "=" at RECORD_TYPE'):
        Host._parse_index_label(FCPath(label), {'method': 'fast'})


def test_a_supplement_label_that_cannot_be_parsed_is_named(tmp_path: Path) -> None:
    label = _write_index(tmp_path, 'vol_index')
    (tmp_path / 'vol_supplemental_index.lbl').write_text(
        'PDS_VERSION_ID = PDS3\nRECORD_TYPE\nEND\n')
    with pytest.raises(PdsHostError,
                       match=r'^cannot parse the PDS3 label '
                             r'.*vol_supplemental_index\.lbl: missing "=" at RECORD_TYPE'):
        Host._index_rows(label)

##########################################################################################
# The supplemental index is optional unless required
##########################################################################################

def test_an_index_without_index_in_its_name_needs_no_supplement(tmp_path: Path) -> None:
    label = _write_index(tmp_path, 'INDEX')
    rows = Host._index_rows(label, supplement=Host.SUPPLEMENT,
                            require_supplement=False)
    assert [row['NUM'] for row in rows] == [1, 2]


def test_a_required_supplement_needs_index_in_the_name(tmp_path: Path) -> None:
    label = _write_index(tmp_path, 'INDEX')
    with pytest.raises(HostIndexError, match='supplemental index required'):
        Host._index_rows(label, supplement=Host.SUPPLEMENT)


def test_a_missing_optional_supplement_is_skipped(tmp_path: Path) -> None:
    label = _write_index(tmp_path, 'vol_index')
    rows = Host._index_rows(label, supplement=Host.SUPPLEMENT,
                            require_supplement=False)
    assert [row['NUM'] for row in rows] == [1, 2]


def test_a_missing_required_supplement_is_not_found(tmp_path: Path) -> None:
    label = _write_index(tmp_path, 'vol_index')
    with pytest.raises(FileNotFoundError, match='vol_supplemental_index.lbl'):
        Host._index_rows(label, supplement=Host.SUPPLEMENT)


def test_a_pointer_to_no_object_is_an_index_error(tmp_path: Path) -> None:
    """A table pointer that names no object of the label is reported as an invalid
    index table, never as a bare KeyError."""

    label = _write_index(tmp_path, 'INDEX')
    label.write_text(label.read_text().replace('OBJECT = INDEX_TABLE', 'OBJECT = TABLE'))
    with pytest.raises(HostIndexError, match="label lacks 'INDEX_TABLE'"):
        Host._index_rows(label)

##########################################################################################
# One host gives WrongHostError; several give UnknownHostError
##########################################################################################

@pytest.mark.parametrize(('writer', 'kind'), FILE_WRITERS)
def test_a_file_rejected_by_one_host_is_the_wrong_host(
        tmp_path: Path, writer: Callable[[Path], Path], kind: str) -> None:
    filepath = writer(tmp_path)
    with pytest.raises(WrongHostError, match=f'not a host_a {kind}: '):
        Host._identify_and_parse(filepath, hosts=[_HostA])


@pytest.mark.parametrize(('writer', 'kind'), FILE_WRITERS)
def test_a_file_rejected_by_several_hosts_is_an_unknown_host(
        tmp_path: Path, writer: Callable[[Path], Path], kind: str) -> None:
    filepath = writer(tmp_path)
    with pytest.raises(UnknownHostError, match=f'unrecognized host in {kind}: '):
        Host._identify_and_parse(filepath, hosts=[_HostA, _HostB])


@pytest.mark.parametrize(('writer', 'kind'), FILE_WRITERS)
def test_a_file_claimed_by_several_hosts_is_ambiguous(
        tmp_path: Path, writer: Callable[[Path], Path], kind: str) -> None:
    filepath = writer(tmp_path)
    with pytest.raises(AmbiguousHostError,
                       match=f'^{kind} .*image.* is claimed by more than one host: '
                             f'claims_all, claims_all_too$'):
        Host._identify_and_parse(filepath, hosts=[_ClaimsAll, _HostA, _ClaimsAllToo],
                                 dispatch=True)


def test_a_label_no_parser_reads_claimed_by_path_by_several_hosts_is_ambiguous(
        tmp_path: Path) -> None:
    label = tmp_path / 'image.lbl'
    label.write_text('PDS_VERSION_ID = PDS3\nRECORD_TYPE\nEND\n')
    with pytest.raises(AmbiguousHostError, match='claims_all, claims_all_too$'):
        Host._identify_and_parse(label, hosts=[_ClaimsAll, _ClaimsAllToo],
                                 pds3_fallback=True, dispatch=True)


def test_a_fits_file_claimed_by_several_hosts_is_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[fits.HDUList] = []
    real_open = fits.open

    def recording_open(*args: object, **kwargs: object) -> fits.HDUList:
        hdulist = real_open(*args, **kwargs)
        opened.append(hdulist)
        return hdulist

    monkeypatch.setattr(fits, 'open', recording_open)
    with pytest.raises(AmbiguousHostError):
        Host._identify_and_parse(_write_fits(tmp_path), hosts=[_ClaimsAll, _ClaimsAllToo])

    assert opened[0]._file.closed


def test_parsed_content_claimed_by_several_hosts_is_ambiguous(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Host, '_HOSTS', [_ClaimsAll, _ClaimsAllToo])
    with pytest.raises(AmbiguousHostError,
                       match='^parsed FITS file is claimed by more than one host: '
                             'claims_all, claims_all_too$'):
        Host._identify_parsed(fits.HDUList([fits.PrimaryHDU()]))


def test_read_fileinfo_considers_only_its_own_host(tmp_path: Path) -> None:
    filepath = _write_vicar(tmp_path)
    with pytest.raises(WrongHostError, match='not a host_a VICAR file: '):
        _HostA._read_fileinfo(filepath, formats='PVFO')

##########################################################################################
