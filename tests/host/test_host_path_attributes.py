##########################################################################################
# tests/host/test_host_path_attributes.py: Host._set_path_attributes names the files
##########################################################################################

from pathlib import Path

import numpy as np
import pytest
from filecache import FCPath

from oops.cadence     import SnapCadence
from oops.fov         import FlatFOV
from oops.host        import Host
from oops.observation import Snapshot


def _snapshot(*, data: bool, header: bool = False) -> Snapshot:
    """A minimal observation, with a data array if `data` is True and a header_dict if
    `header` is True."""

    subfields: dict[str, object] = {'data': np.zeros((2, 2))} if data else {}
    if header:
        subfields['header_dict'] = {}
    return Snapshot(('v', 'u'), cadence=SnapCadence(0., 1.),
                    fov=FlatFOV((1.e-3, 1.e-3), (2, 2)), path='SSB', frame='J2000',
                    **subfields)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    filepath = tmp_path / 'data' / 'IMAGE.IMG'
    filepath.parent.mkdir()
    filepath.write_bytes(b'')
    return filepath


@pytest.fixture
def label(tmp_path: Path) -> Path:
    filepath = tmp_path / 'data' / 'IMAGE.LBL'
    filepath.parent.mkdir(exist_ok=True)
    filepath.write_text('END\n')
    return filepath

##########################################################################################
# An image file passed alone
##########################################################################################

def test_an_image_alone_gives_its_path_name_url_and_local_path_and_no_label(
        image: Path) -> None:
    obs = _snapshot(data=False)
    Host._set_path_attributes(obs, image, None)
    expected = {'filepath': FCPath(image), 'basename': 'IMAGE.IMG',
                'image_url': image.as_posix(), 'image_abspath': image.resolve()}

    assert obs.subfields == expected

##########################################################################################
# A label passed
##########################################################################################

def test_a_label_gives_the_image_and_label_subfields(image: Path, label: Path) -> None:
    obs = _snapshot(data=True)
    Host._set_path_attributes(obs, image, label)
    expected = {'filepath': FCPath(image), 'basename': 'IMAGE.IMG',
                'image_url': image.as_posix(), 'image_abspath': image.resolve(),
                'label_url': label.as_posix(), 'label_abspath': label.resolve()}

    assert {key: obs.subfields[key] for key in expected} == expected


def test_a_label_without_the_image_array_gives_the_image_url_but_no_image_abspath(
        image: Path, label: Path) -> None:
    """For astrometry, the host reads the label and leaves the image file unread."""

    obs = _snapshot(data=False)
    Host._set_path_attributes(obs, image, label)
    expected = {'filepath': FCPath(image), 'basename': 'IMAGE.IMG',
                'image_url': image.as_posix(), 'label_url': label.as_posix(),
                'label_abspath': label.resolve()}

    assert obs.subfields == expected


def test_a_label_with_the_header_of_the_image_file_gives_image_abspath(
        image: Path, label: Path) -> None:
    obs = _snapshot(data=False, header=True)
    Host._set_path_attributes(obs, image, label)

    assert obs.image_abspath == image.resolve()

##########################################################################################
# An index row
##########################################################################################

def test_an_index_row_gives_only_its_relative_path_and_name() -> None:
    obs = _snapshot(data=False)
    Host._set_path_attributes(obs, 'COISS_2001/data/1/N1_1.IMG', None)

    assert obs.subfields == {'filepath': 'COISS_2001/data/1/N1_1.IMG',
                             'basename': 'N1_1.IMG'}

##########################################################################################
