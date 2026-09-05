##########################################################################################
# oops/hosts/juno/jiram/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.jiram`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from numpy import ndarray, number
from polymath import Scalar

# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `str` is excluded deliberately: no polymath constructor accepts one.
_Numeric = float | number | list['_Numeric'] | tuple['_Numeric', ...]
ScalarLike = Scalar | ndarray | _Numeric

__all__ = ['from_file', 'JIRAM']

def from_file(filespec: Any, return_all_planets: bool = False, method: str = 'strict',
    **parameters: Any) -> Any: ...

class JIRAM:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    @staticmethod
    def initialize(asof: str | None = None, **kwargs: Any) -> None: ...
    @staticmethod
    def create_frame(time: ScalarLike, name: str) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
