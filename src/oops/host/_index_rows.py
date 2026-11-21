##########################################################################################
# oops/host/_index_rows.py: reading, pairing and filtering the rows of an index
##########################################################################################
"""Support functions for :meth:`oops.Host._index_rows`.

An index and its supplement are read with :class:`~pdstable.PdsTable` a few rows at a
time, so that a host can build one observation per row while holding only a few rows in
memory.
"""

import itertools

import numpy as np
from pdsparser import PdsError
from pdstable  import PdsTable

from oops._exceptions import OopsTypeError, OopsValueError

from ._errors import HostIndexError, PdsHostError

# The number of rows that PdsTable reads and parses at a time. Pairing the 464,752 rows of
# COVIMS_0999 with its supplement took 26.8 s at 1,000 rows, 30.2 s at 100, 43.2 s at 20,
# and 59.7 s at 10; reading the index alone one row at a time took 198 s. A host takes
# about 1.5 ms to build an observation from a row, so 100 rows instead of 1,000 adds about
# 0.5% to the time of a pass, and a row longer than the label says holds back at most the
# 99 rows read with it.
CHUNK_ROWS = 100

# The number of paths that an error message about unused supplement rows lists
_PATHS_IN_MESSAGE = 3

# The marker for the end of a table
_END = object()


def open_table(filepath, *, label=None, pds3_method='fast'):
    """A PDS3 table whose label has been read and whose rows have not.

    Parameters:
        filepath (FCPath): The path to the label of the table.
        label (Pds3Label | dict[str, Any], optional): The content of the label, if it
            has already been read.
        pds3_method (str, optional): The method of the PDS3 label parser.

    Returns:
        PdsTable: The table, created with ``defer_load=True``.

    Raises:
        HostIndexError: If the label does not describe a table that can be read, as when
            it lacks the object that its table pointer refers to.
        OSError: If the label does not exist or cannot be read.
        PdsHostError: If the label uses invalid PDS3 syntax. The message has the form
            "cannot parse the PDS3 label <filepath>: <parser's message>".
    """

    try:
        return PdsTable(filepath, label_contents=label, label_method=pds3_method,
                        defer_load=True)
    except PdsError as err:
        raise PdsHostError(f'cannot parse the PDS3 label {filepath}: {err}') from err
    except KeyError as err:
        # An item that the table reader needs is not in the label, such as the object
        # that a pointer refers to (^INDEX_TABLE in a label whose table object is TABLE)
        raise HostIndexError(f'invalid index table {filepath}: the label lacks '
                             f'{err}') from err
    except (TypeError, ValueError) as err:
        # The PDS3 options were checked when the label was parsed, so the problem lies in
        # the description of the table
        raise HostIndexError(f'invalid index table {filepath}: {err}') from err


def stream_rows(table, filepath, chunk_rows=None):
    """Yield the rows of a table, reading them a few at a time as the iteration proceeds.

    Parameters:
        table (PdsTable): The table, created with ``defer_load=True``.
        filepath (FCPath): The path to the label of the table, for error messages.
        chunk_rows (int, optional): The number of rows to read and parse at a time; by
            default, ``CHUNK_ROWS``.

    Yields:
        dict[str, Any]: One dictionary per row, as :meth:`pdstable.PdsTable.iter_dicts`
        gives it.

    Raises:
        HostIndexError: When the iteration reaches the end of a file that holds fewer
            rows than its label says, after every row it holds; at the first row of a
            chunk that holds a row longer than the label's row length; or, after the last
            row, when the file holds more rows than its label says. A value that cannot be
            read as its column's number only makes PdsTable issue a UserWarning, and the
            row is yielded with the value as a string.
        OSError: When the iteration starts, if the table file does not exist or cannot be
            read.
    """

    rows = table.iter_dicts(chunk_rows or CHUNK_ROWS)
    while True:
        try:
            row = next(rows)
        except StopIteration:
            return
        except ValueError as err:
            raise HostIndexError(f'invalid index table {filepath}: {err}') from err
        yield row


def first_row(table, filepath):
    """The first row of a table, read on its own.

    Parameters:
        table (PdsTable): The table, created with ``defer_load=True``.
        filepath (FCPath): The path to the label of the table, for error messages.

    Returns:
        dict[str, Any] | None: The first row; None if the table has no rows.

    Raises:
        HostIndexError: If the first row is longer than the label's row length.
        OSError: If the table file does not exist or cannot be read.
    """

    rows = stream_rows(table, filepath, chunk_rows=1)
    try:
        return next(rows, None)
    finally:
        rows.close()


def column_names(table):
    """Every key that a row dictionary of a table can have.

    Parameters:
        table (PdsTable): The table.

    Returns:
        set[str]: Each column name, the name with each blank replaced by an underscore,
        and each of those followed by "_mask".
    """

    names = set()
    for name in table.get_keys():
        names |= {name, name.replace(' ', '_')}
    return names | {name + '_mask' for name in names}


def key_columns(*keys):
    """The names of the columns that make up one or more file keys.

    Parameters:
        *keys (str | tuple[str, ...] | collections.abc.Callable): The keys.

    Returns:
        set[str]: Every column that a key lists; a function lists none.
    """

    names = set()
    for key in keys:
        if isinstance(key, str):
            names.add(key)
        elif isinstance(key, (list, tuple)):
            names.update(key)
    return names


def row_path(row, key):
    """The path of the file that a row gives, as the table writes it.

    Parameters:
        row (dict[str, Any]): The row.
        key (str | tuple[str, ...] | collections.abc.Callable): A column name, a tuple of
            column names whose values are joined with "/", or a function that returns the
            path from a row dictionary.

    Returns:
        str: The path.
    """

    if callable(key):
        return key(row)
    names = (key,) if isinstance(key, str) else key
    return '/'.join(str(row[name]) for name in names)


def path_key(path):
    """The form of a path in which an index and its supplement are compared.

    Parameters:
        path (str): A path as a table writes it.

    Returns:
        str: The path in upper case, without empty or "." components or the extension of
        the file name, so that "./data/a.lbl" and "/DATA/A.QUB" have the same key.
    """

    parts = [part for part in path.strip().upper().split('/') if part not in ('', '.')]
    if parts:
        parts[-1] = parts[-1].rpartition('.')[0] or parts[-1]
    return '/'.join(parts)


def column_validator(names):
    """A function that checks a pair of rows by comparing the values of named columns.

    Parameters:
        names (list[str]): The columns to compare.

    Returns:
        collections.abc.Callable: A function of an index row and a supplement row that
        returns True if every named column has the same value in both, and otherwise
        raises HostIndexError, whose message gives the column. Two strings are the same
        if either one, in upper case, contains the other.
    """

    def validator(index_row, supplement_row):
        for name in names:
            index_value = index_row[name]
            supplement_value = supplement_row[name]
            if index_value == supplement_value:
                continue
            if isinstance(index_value, str) and isinstance(supplement_value, str):
                index_upper = index_value.upper()
                supplement_upper = supplement_value.upper()
                if supplement_upper in index_upper or index_upper in supplement_upper:
                    continue
            raise HostIndexError(f'{name} value mismatch: {_plain(index_value)!r}, '
                                 f'{_plain(supplement_value)!r}')
        return True

    return validator


def check_filter(filter, names):
    """Raise OopsValueError if a filter uses a column that the rows do not have.

    Parameters:
        filter (dict[str, Any] | None): The filter.
        names (collections.abc.Container[str]): The keys of the row dictionaries.

    Raises:
        OopsValueError: If a key of `filter` is not in `names`.
    """

    for key in filter or {}:
        if key not in names:
            raise OopsValueError(f'filter uses a column that the index does not have: '
                                 f'{key!r}')


def passes_filter(row, filter):
    """True if a row passes a filter.

    Parameters:
        row (dict[str, Any]): The row.
        filter (dict[str, Any]): Column names, each with a value, a set, list, or tuple of
            values, or a tuple of two values that bound a range. A row passes if every
            column it lists equals the value, is one of the values, or lies within the
            range, bounds included. A row that lacks one of the columns fails.

    Returns:
        bool: True if the row passes.

    Raises:
        OopsTypeError: If a range cannot be compared with the row's value.
    """

    for (key, values) in filter.items():
        if key not in row:
            return False
        value = row[key]
        if isinstance(values, tuple) and len(values) == 2:
            # Tested as inside rather than outside, so that a NaN, which compares false
            # with every bound, fails
            try:
                inside = values[0] <= value <= values[1]
            except TypeError as err:
                raise OopsTypeError(f'filter range {values!r} cannot be compared with '
                                    f'column {key!r} value {value!r}') from err
            if not inside:
                return False
        elif isinstance(values, (set, frozenset, list, tuple)):
            if value not in values:
                return False

        # A single value; "in" would do a substring match on strings
        elif value != values:
            return False

    return True


def filtered(rows, filter, names=None):
    """Yield the rows that pass a filter, testing each when the iteration reaches it.

    Parameters:
        rows (collections.abc.Iterable[dict[str, Any]]): The rows.
        filter (dict[str, Any] | None): The filter, as for :func:`passes_filter`; None or
            empty to pass every row.
        names (collections.abc.Container[str] | None, optional): The keys of the rows, if
            they were checked against `filter` already; None to check `filter` against
            the keys of the first row when it is read.

    Yields:
        dict[str, Any]: Each row that passes.

    Raises:
        OopsTypeError: When the iteration reaches a row whose value cannot be compared
            with a range in `filter`.
        OopsValueError: When the first row is read, if `names` is None and `filter` uses
            a column that the first row does not have.
    """

    if not filter:
        yield from rows
        return

    for row in rows:
        if names is None:
            check_filter(filter, row)
            names = row
        if passes_filter(row, filter):
            yield row


def _plain(value):
    """A NumPy scalar as the Python value it holds, for a message; any other value as it
    is."""

    return value.item() if isinstance(value, np.generic) else value


def _merge(index_row, supplement_row, prefer_supplement):
    """One index row merged with its supplement row."""

    if prefer_supplement:
        return index_row | supplement_row
    return supplement_row | index_row


def _check_pair(validate, index_row, supplement_row, i, files):
    """Raise HostIndexError if a pair of rows fails a check.

    Parameters:
        validate (collections.abc.Callable | None): The check; None for no check.
        index_row (dict[str, Any]): The index row.
        supplement_row (dict[str, Any]): The supplement row.
        i (int): The number of the index row, counting from 0, for the message.
        files (str): The paths to the index and the supplement, for the message.

    Raises:
        HostIndexError: If `validate` returns False or raises an exception.
    """

    if validate is None:
        return
    try:
        status = validate(index_row, supplement_row)
    except Exception as err:
        raise HostIndexError(f'{err}\nindex row {i} does not match its supplement row: '
                             f'{files}') from err
    if not status:
        raise HostIndexError(f'index row {i} does not match its supplement row: {files}')


def pair_by_position(index_rows, supplement_rows, *, file_key, supplement_key, validate,
                     prefer_supplement, files):
    """Yield each index row merged with the supplement row in the same position.

    Both tables are read in step, one row of each at a time.

    Parameters:
        index_rows (collections.abc.Iterator[dict[str, Any]]): The index rows.
        supplement_rows (collections.abc.Iterator[dict[str, Any]]): The supplement rows.
        file_key (str | tuple[str, ...] | collections.abc.Callable): How an index row
            gives its file.
        supplement_key (str | tuple[str, ...] | collections.abc.Callable): How a
            supplement row gives its file.
        validate (collections.abc.Callable | None): The check of each pair; None for none.
        prefer_supplement (bool): True to keep the supplement's value of a column that
            both tables have; False to keep the index's.
        files (str): The paths to the index and the supplement, for error messages.

    Yields:
        dict[str, Any]: Each merged row, in index order.

    Raises:
        HostIndexError: At an index row, counting from 0, whose file is not the file of
            the supplement row in the same position, which fails `validate`, or for which
            the supplement has ended; after the last index row, if the supplement has
            more rows.
    """

    for (i, index_row) in enumerate(index_rows):
        supplement_row = next(supplement_rows, _END)
        if supplement_row is _END:
            raise HostIndexError(f'the supplement ends before index row {i}: {files}')
        index_path = row_path(index_row, file_key)
        supplement_path = row_path(supplement_row, supplement_key)
        if path_key(index_path) != path_key(supplement_path):
            raise HostIndexError(f'index row {i} gives {index_path!r} but the '
                                 f'supplement row in the same position gives '
                                 f'{supplement_path!r}; the rows of this host\'s '
                                 f'tables are paired by position: {files}')
        _check_pair(validate, index_row, supplement_row, i, files)
        yield _merge(index_row, supplement_row, prefer_supplement)

    if next(supplement_rows, _END) is not _END:
        raise HostIndexError(f'the supplement has more rows than the index: {files}')


def pair_by_file(index_rows, supplement_rows, *, file_key, supplement_key, validate,
                 prefer_supplement, files):
    """Yield each index row merged with the supplement row that gives the same file.

    The index is read as the iteration proceeds. The supplement is read ahead only as far
    as the partner of the current index row; each supplement row read before its partner
    is held until that partner is reached, and then released.

    Parameters:
        index_rows (collections.abc.Iterator[dict[str, Any]]): The index rows.
        supplement_rows (collections.abc.Iterator[dict[str, Any]]): The supplement rows.
        file_key (str | tuple[str, ...] | collections.abc.Callable): How an index row
            gives its file.
        supplement_key (str | tuple[str, ...] | collections.abc.Callable): How a
            supplement row gives its file.
        validate (collections.abc.Callable | None): The check of each pair; None for none.
        prefer_supplement (bool): True to keep the supplement's value of a column that
            both tables have; False to keep the index's.
        files (str): The paths to the index and the supplement, for error messages.

    Yields:
        dict[str, Any]: Each merged row, in index order.

    Raises:
        HostIndexError: At an index row, counting from 0, that fails `validate`; at a
            supplement row that repeats the file of a supplement row that is still held;
            at an index row whose file the supplement does not list, once the whole
            supplement has been read, which is also where an index that lists a file
            more often than the supplement fails; after the last index row, if any
            supplement row was not paired with an index row, which is also where a
            supplement that lists a file more often than the index fails.
    """

    # Only the rows read ahead are kept. A set of every file already paired would find a
    # repeated file sooner, but it would grow with the index, by about 60 MB for the
    # 464,752 rows of COVIMS_0999; a repeat still raises, at the end of the supplement.
    held = {}           # path key -> (path, supplement row), read but not yet paired
    supplement_rows = enumerate(supplement_rows)

    for (i, index_row) in enumerate(index_rows):
        index_path = row_path(index_row, file_key)
        key = path_key(index_path)
        while key not in held:
            (j, supplement_row) = next(supplement_rows, (None, _END))
            if supplement_row is _END:
                raise HostIndexError(f'the supplement does not list the file of index '
                                     f'row {i}, {index_path!r}, or lists it fewer times '
                                     f'than the index: {files}')
            supplement_path = row_path(supplement_row, supplement_key)
            supplement_key_ = path_key(supplement_path)
            if supplement_key_ in held:
                raise HostIndexError(f'the supplement gives {supplement_path!r} a second '
                                     f'time, at row {j}: {files}')
            held[supplement_key_] = (supplement_path, supplement_row)

        supplement_row = held.pop(key)[1]
        _check_pair(validate, index_row, supplement_row, i, files)
        yield _merge(index_row, supplement_row, prefer_supplement)

    unpaired = [path for (path, _) in held.values()]
    unpaired += [row_path(row, supplement_key) for (_, row) in
                 itertools.islice(supplement_rows, 1)]
    if unpaired:
        raise HostIndexError(f'the supplement lists files that the index does not, or '
                             f'lists them more times, such as '
                             f'{unpaired[:_PATHS_IN_MESSAGE]}: {files}')

##########################################################################################
