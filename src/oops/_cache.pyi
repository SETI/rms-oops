##########################################################################################
# oops/_cache.pyi
##########################################################################################
"""Type stub for :mod:`oops._cache`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.oops import Oops as Oops

class _Cache(Oops):
    def __init__(self, maxsize: int = 100) -> None: ...
    def __len__(self) -> int: ...
    @staticmethod
    def clean_key(key: Any) -> Any: ...
    def __contains__(self, key: Any) -> bool: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

##########################################################################################
