##########################################################################################
# oops/hosts/juno/junocam/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.junocam`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from filecache import FCPath as FCPath
from oops import Path as Path

__all__ = ['from_file', 'JUNOCAM']

def from_file(filespec: str | Path | FCPath, fast_distortion: bool = True,
    return_all_planets: bool = False, snap: bool = False, method: str = 'strict',
    **parameters: Any) -> Any: ...

class JUNOCAM:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    @staticmethod
    def initialize(asof: str | None = None, **kwargs: Any) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
