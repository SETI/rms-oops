##########################################################################################
# tests/host/test_host_file_paths.py: the file subfields that Host._apply_overrides sets
##########################################################################################

from pathlib import Path
from typing import Any, cast

from filecache import FCPath, FileCache

from oops.host import Host

# A URL that nothing can retrieve and that needs no cloud credentials, so that the tests
# depend on neither the network nor the account of the machine that runs them
URL = 'https://rms-test.invalid/volume/data/image.img'


def _paths(filepath: Any, **kwargs: Any) -> dict[str, Any]:
    """The parameters that Host._apply_overrides sets to name a file."""

    return Host._apply_overrides({}, filepath, **kwargs)


def _remote(tmp_path: Path) -> FCPath:
    """A remote path whose local cache is private to the test."""

    cache = FileCache('paths', cache_root=tmp_path / 'cache', delete_on_exit=True)
    return FCPath(URL, filecache=cache)


def test_an_existing_local_file_has_no_abspath_unless_it_is_given(tmp_path: Path) -> None:
    filepath = tmp_path / 'image.img'
    filepath.write_bytes(b'')

    assert 'abspath' not in _paths(FCPath(filepath))


def test_a_cached_remote_file_has_no_abspath_unless_it_is_given(tmp_path: Path) -> None:
    filepath = _remote(tmp_path)
    cast(Path, filepath.get_local_path()).write_bytes(b'')

    assert 'abspath' not in _paths(filepath)


def test_a_remote_file_creates_no_cache_directory(tmp_path: Path) -> None:
    filepath = _remote(tmp_path)
    _paths(filepath)

    local_path = cast(Path, filepath.get_local_path(create_parents=False))
    assert not local_path.parent.exists()


def test_the_abspath_given_is_set_resolved(tmp_path: Path) -> None:
    local_path = tmp_path / 'cache' / '..' / 'image.img'

    assert _paths(_remote(tmp_path), abspath=local_path)['abspath'] == \
        (tmp_path / 'image.img').resolve()


def test_a_remote_file_has_its_url_as_image_url(tmp_path: Path) -> None:
    assert _paths(_remote(tmp_path))['image_url'] == URL


def test_a_string_names_the_file_without_a_local_path() -> None:
    assert _paths('VOLUME/DATA/IMAGE.IMG') == {'filepath': 'VOLUME/DATA/IMAGE.IMG',
                                               'basename': 'IMAGE.IMG'}


def test_no_path_names_no_file() -> None:
    assert _paths(None) == {}

##########################################################################################
