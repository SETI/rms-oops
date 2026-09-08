##########################################################################################
# oops/hosts/juno/junocam/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.junocam`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

from pathlib import Path
from typing import Any
from filecache import FCPath as FCPath
from oops import Observation as Observation

__all__ = ['from_file', 'JUNOCAM']

def from_file(filespec: str | Path | FCPath, fast_distortion: bool | None = True,
    return_all_planets: bool = False, snap: bool = False, method: str = 'strict',
    **parameters: Any) -> list[Observation]: ...

class JUNOCAM:
    instrument_kernel: dict | None
    fovs: dict
    initialized: bool
    @staticmethod
    def initialize(asof: str | None = None, **kwargs: Any) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
