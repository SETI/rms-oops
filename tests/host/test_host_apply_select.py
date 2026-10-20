##########################################################################################
# tests/host/test_host_apply_select.py: The select option of Host._apply_select
##########################################################################################

from typing import Any

import numpy as np
import pytest

from oops         import OopsIndexError, OopsTypeError, OopsValueError
from oops.host    import Host

NAMES = {'ALPHA': 0, 'BETA': 1, 'GAMMA': 2}


class _Obs:
    """A stand-in for an Observation, which _apply_select only stores and returns."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f'_Obs({self.label!r})'


@pytest.fixture
def flat() -> tuple[_Obs, ...]:
    """Three observations from one file, as a tuple."""

    return (_Obs('a'), _Obs('b'), _Obs('c'))


@pytest.fixture
def grid() -> np.ndarray[Any, np.dtype[np.object_]]:
    """A 2x2 object array of observations, labeled by row and column."""

    array = np.empty((2, 2), dtype=np.object_)
    for i in range(2):
        for j in range(2):
            array[i, j] = _Obs(f'{i}{j}')
    return array


def test_none_returns_the_observations_themselves(flat: tuple[_Obs, ...]) -> None:
    assert Host._apply_select(flat, None) is flat


@pytest.mark.parametrize(('select', 'index'), [(0, 0), (2, 2), (-1, 2), (np.int64(1), 1)])
def test_an_integer_chooses_one_observation(flat: tuple[_Obs, ...], select: Any,
                                            index: int) -> None:
    assert Host._apply_select(flat, select) is flat[index]


def test_a_slice_of_a_tuple_is_a_tuple(flat: tuple[_Obs, ...]) -> None:
    assert Host._apply_select(flat, slice(1, None)) == flat[1:]


@pytest.mark.parametrize('select', ['beta', 'BETA', 'Beta'])
def test_a_name_ignores_case(flat: tuple[_Obs, ...], select: str) -> None:
    assert Host._apply_select(flat, select, NAMES) is flat[1]


def test_an_integer_out_of_range_is_an_index_error(flat: tuple[_Obs, ...]) -> None:
    with pytest.raises(OopsIndexError, match='select index out of range: 3'):
        Host._apply_select(flat, 3)


def test_an_unknown_name_is_a_value_error(flat: tuple[_Obs, ...]) -> None:
    with pytest.raises(OopsValueError, match="unrecognized select name: 'delta'"):
        Host._apply_select(flat, 'delta', NAMES)


def test_a_name_without_names_is_a_value_error(flat: tuple[_Obs, ...]) -> None:
    with pytest.raises(OopsValueError, match="unrecognized select name: 'beta'"):
        Host._apply_select(flat, 'beta')


@pytest.mark.parametrize('select', [True, 1.5, [0, 1]], ids=['bool', 'float', 'list'])
def test_an_invalid_select_is_a_type_error(flat: tuple[_Obs, ...], select: Any) -> None:
    with pytest.raises(OopsTypeError, match='invalid select'):
        Host._apply_select(flat, select)


def test_a_full_tuple_index_of_an_array_chooses_one_observation(
        grid: np.ndarray[Any, np.dtype[np.object_]]) -> None:
    assert Host._apply_select(grid, (0, 0)) is grid[0, 0]
    assert Host._apply_select(grid, (1, -1)) is grid[1, 1]


def test_a_tuple_with_a_slice_indexes_like_numpy(
        grid: np.ndarray[Any, np.dtype[np.object_]]) -> None:
    column = Host._apply_select(grid, (slice(None), 0))
    assert isinstance(column, np.ndarray)
    assert list(column) == [grid[0, 0], grid[1, 0]]


def test_a_short_tuple_index_of_an_array_chooses_a_row(
        grid: np.ndarray[Any, np.dtype[np.object_]]) -> None:
    row = Host._apply_select(grid, (1,))
    assert isinstance(row, np.ndarray)
    assert list(row) == [grid[1, 0], grid[1, 1]]


def test_a_tuple_index_of_a_tuple_chooses_one_observation(
        flat: tuple[_Obs, ...]) -> None:
    assert Host._apply_select(flat, (2,)) is flat[2]


def test_a_tuple_slice_of_a_tuple_is_a_tuple(flat: tuple[_Obs, ...]) -> None:
    assert Host._apply_select(flat, (slice(None, 2),)) == flat[:2]


def test_a_name_in_a_tuple_stands_for_its_index(
        grid: np.ndarray[Any, np.dtype[np.object_]]) -> None:
    assert Host._apply_select(grid, ('alpha', 'beta'), NAMES) is grid[0, 1]


def test_too_many_tuple_indices_is_an_index_error(
        grid: np.ndarray[Any, np.dtype[np.object_]]) -> None:
    with pytest.raises(OopsIndexError, match=r'invalid select \(0, 0, 0\)'):
        Host._apply_select(grid, (0, 0, 0))


def test_a_tuple_index_out_of_range_is_an_index_error(flat: tuple[_Obs, ...]) -> None:
    with pytest.raises(OopsIndexError, match='index 5 is out of bounds'):
        Host._apply_select(flat, (5,))


def test_an_unknown_name_in_a_tuple_is_a_value_error(
        grid: np.ndarray[Any, np.dtype[np.object_]]) -> None:
    with pytest.raises(OopsValueError, match="unrecognized select name: 'delta'"):
        Host._apply_select(grid, (0, 'delta'), NAMES)


@pytest.mark.parametrize('item', [(0,), True, 1.5], ids=['tuple', 'bool', 'float'])
def test_an_invalid_tuple_item_is_a_type_error(
        grid: np.ndarray[Any, np.dtype[np.object_]], item: Any) -> None:
    with pytest.raises(OopsTypeError, match='invalid select item'):
        Host._apply_select(grid, (0, item))

##########################################################################################
