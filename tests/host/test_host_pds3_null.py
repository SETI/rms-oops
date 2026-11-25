##########################################################################################
# tests/host/test_host_pds3_null.py: is_pds3_null and PDS3_NULL_TEXTS
##########################################################################################

from typing import Any

import pytest

from oops.host import PDS3_NULL_TEXTS, is_pds3_null


def test_the_null_texts_are_the_seven_placeholders() -> None:
    assert PDS3_NULL_TEXTS == {'UNK', 'N/A', 'NULL', 'UNKNOWN', 'NOT APPLICABLE', '---',
                               ''}


@pytest.mark.parametrize('value', ['UNK', 'n/a', '  Null ', 'unknown', 'Not Applicable',
                                   '---', '', '   '])
def test_a_null_text_in_any_case_and_padding_is_null(value: str) -> None:
    assert is_pds3_null(value)


@pytest.mark.parametrize('value', ['UNKX', 'NOT UNK', 'N / A', '--', 'NONE'],
                         ids=['longer', 'contains', 'inner-blanks', 'shorter', 'none'])
def test_a_text_that_is_not_a_whole_null_text_is_not_null(value: str) -> None:
    assert not is_pds3_null(value)


@pytest.mark.parametrize('value', [None, 0, -999, ['UNK'], ('UNK',)],
                         ids=['none', 'zero', 'fill', 'list', 'tuple'])
def test_a_value_that_is_not_a_string_is_not_null(value: Any) -> None:
    assert not is_pds3_null(value)

##########################################################################################
