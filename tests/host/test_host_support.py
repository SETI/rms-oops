##########################################################################################
# tests/host/test_host_support.py: the argument checks and index paths of Host
##########################################################################################

import pytest
from filecache import FCPath

from oops      import OopsTypeError, OopsValueError
from oops.host import Host


class _Named(Host):
    """A host whose class name appears in messages."""

    NAME = 'named'


@pytest.mark.parametrize(('name', 'expected'),
                         [('VOL_0001_index.lbl', 'VOL_0001_supplemental_index.lbl'),
                          ('VOL_0001_INDEX.LBL', 'VOL_0001_SUPPLEMENTAL_INDEX.LBL')],
                         ids=['lower', 'upper'])
def test_the_supplement_is_named_after_the_index(name: str, expected: str) -> None:
    assert Host._supplement_path(FCPath('/data') / name) == FCPath('/data') / expected


def test_an_index_without_index_in_its_name_has_no_supplement() -> None:
    assert Host._supplement_path(FCPath('/data/table.lbl')) is None


@pytest.mark.parametrize(('index', 'expected'),
                         [('/h/metadata/GO_0xxx/GO_0017/GO_0017_index.lbl',
                           '/h/volumes/GO_0xxx'),
                          ('/h/METADATA/GO_0xxx/GO_0017/GO_0017_index.lbl',
                           '/h/volumes/GO_0xxx'),
                          ('/elsewhere/GO_0017_index.lbl', '/elsewhere')],
                         ids=['metadata', 'upper-case', 'elsewhere'])
def test_the_volumes_directory_follows_the_location_of_the_index(index: str,
                                                                 expected: str) -> None:
    assert Host._volumes_directory(index) == FCPath(expected)


def test_an_unexpected_keyword_raises_naming_the_class_and_method() -> None:
    with pytest.raises(OopsTypeError, match="unexpected keyword argument for "
                                            "_Named.from_index: 'supplementary'"):
        _Named._check_keywords('from_index', {'supplementary': 1})


def test_select_must_be_none_for_one_observation_per_row() -> None:
    with pytest.raises(OopsValueError, match='each row of an index is one observation; '
                                             'select must be None, not 0'):
        Host._check_select(0, unit='row of an index')


def test_select_with_an_unknown_name_raises() -> None:
    with pytest.raises(OopsValueError, match="unrecognized select name: 'VIS'"):
        Host._check_select(('IR', 'VIS'), names={'IR': 0})


def test_select_of_another_type_raises() -> None:
    with pytest.raises(OopsTypeError, match='invalid select: 1.5'):
        Host._check_select(1.5)


def test_a_valid_select_passes() -> None:
    assert Host._check_select((slice(None), 'ir', 0), names={'IR': 0}) is None

##########################################################################################
