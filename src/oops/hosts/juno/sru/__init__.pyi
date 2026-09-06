##########################################################################################
# oops/hosts/juno/sru/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.sru`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from filecache import FCPath as FCPath
from oops import Frame as Frame, Path as Path
from numpy import ndarray, number
from polymath import Scalar
# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `polymath.typedefs` names each of those unions.
from polymath.typedefs import ScalarLike
from oops.hosts.juno import Juno as Juno

class SRU:
    SAMPLES: int
    LINES: int
    UV_LOS: Any
    FL_PIXELS: float
    DISTORTION: Any
    spice_frames: Any
    initialized: bool
    @staticmethod
    def initialize(asof: str | None = None, **kwargs: Any) -> None: ...
    @staticmethod
    def fov() -> Any: ...
    @staticmethod
    def create_frame(unit: Any, time: ScalarLike) -> Frame: ...
    @staticmethod
    def reset() -> None: ...

def from_file(filespec: str | Path | FCPath, return_all_planets: bool = False,
    method: str = 'strict', **parameters: Any) -> Any: ...

##########################################################################################
