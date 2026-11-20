##########################################################################################
# tests/host/test_host_filter_rows.py: Host._filter_index_rows
##########################################################################################

import pytest

from oops      import OopsTypeError, OopsValueError
from oops.host import Host

ROWS = [{'NAME': 'A', 'NUM': 1}, {'NAME': 'B', 'NUM': 2}, {'NAME': 'C'}]


def test_an_unknown_column_raises_an_error_naming_it() -> None:
    with pytest.raises(OopsValueError, match="'NOPE'"):
        Host._filter_index_rows(ROWS, {'NOPE': 1})


def test_a_row_without_the_column_is_left_out() -> None:
    """A row that lacks a column, as when its value is missing, fails the filter."""

    rows = Host._filter_index_rows(ROWS, {'NUM': (0, 9)})

    assert [row['NAME'] for row in rows] == ['A', 'B']


def test_a_range_that_cannot_be_compared_raises() -> None:
    with pytest.raises(OopsTypeError, match="column 'NUM'"):
        Host._filter_index_rows(ROWS, {'NUM': ('a', 'z')})

##########################################################################################
