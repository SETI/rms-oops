##########################################################################################
# spicedb/sqlite_db.pyi
##########################################################################################
"""Type stub for :mod:`spicedb.sqlite_db`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

CONNECTION: Any
CURSOR: Any

def open(filepath: Any) -> None: ...

def close() -> None: ...

def query(sql_string: Any) -> Any: ...

##########################################################################################
