##########################################################################################
# oops/hosts/juno/jiram/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.jiram`.

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
from numpy import ndarray, number
from oops.observation import Slit1D as Slit1D, Snapshot as Snapshot
from polymath import Scalar
# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `polymath.typedefs` names each of those unions.
from polymath.typedefs import ScalarLike

__all__ = ['from_file', 'JIRAM']

def from_file(filespec: str | Path | FCPath, return_all_planets: bool = False,
    method: str = 'strict',
    **parameters: Any) -> list[Snapshot] | tuple[Slit1D, list[Snapshot]] | None: ...

class JIRAM:
    instrument_kernel: dict | None
    fovs: dict
    initialized: bool
    @staticmethod
    def initialize(asof: str | None = None, **kwargs: Any) -> None: ...
    @staticmethod
    def create_frame(time: ScalarLike, name: str) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
