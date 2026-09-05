##########################################################################################
# oops/lightsource/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.lightsource`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from oops import Body as Body, Event as Event, RPD as RPD
from numpy import ndarray, number
from polymath import Pair, Scalar, Vector3
from oops.oops import Oops as Oops
from oops.path import Path as Path

# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `str` is excluded deliberately: no polymath constructor accepts one.
_Numeric = float | number | list['_Numeric'] | tuple['_Numeric', ...]
ScalarLike = Scalar | ndarray | _Numeric

PairLike = Pair | ndarray | _Numeric
Vector3Like = Vector3 | ndarray | _Numeric

__all__ = ['LightSource', 'DiskSource']

class LightSource(Oops):
    name: Any
    source: Any
    source_is_moving: bool
    shape: Any
    weight: Any
    def __init__(self, name: str, source: Path | str | PairLike | Vector3Like,
        weight: ScalarLike | None = None) -> None: ...
    def photon_to_event(self, event: Event, derivs: bool = False,
        guess: ScalarLike | None = None, antimask: ndarray | bool | None = None,
        quick: dict | bool | None = None,
        converge: dict | None = None) -> tuple[Event | None, Event]: ...
    def as_path(self) -> Any: ...

class DiskSource(LightSource):
    name: Any
    source: Any
    source_is_moving: Any
    xy_grid: Any
    shape: Any
    radius: Any
    weight: Any
    def __init__(self, name: str, source: Path | tuple | Vector3Like, radius: float,
        size: int = 11, compress: bool = False) -> None: ...
    def photon_to_event(self, event: Event, derivs: bool = False,
        guess: ScalarLike | None = None, antimask: ndarray | bool | None = None,
        quick: dict | bool | None = None,
        converge: dict | None = None) -> tuple[None, Event]: ...

##########################################################################################
