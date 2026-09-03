##########################################################################################
# oops/surface/ringplane.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.ringplane`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.constants import TWOPI as TWOPI
from oops.frame.frame_ import Frame as Frame
from oops.gravity.oblategravity import OblateGravity as OblateGravity
from oops.path.path_ import Path as Path
from oops.surface.surface_ import Surface as Surface

class RingPlane(Surface):
    COORDINATE_TYPE: str
    COORDINATE_NAMES: Any
    COORDINATE_ABBREVS: Any
    COORDINATE_RANGES: Any
    IS_VIRTUAL: bool
    IS_TIME_DEPENDENT: bool
    origin: Any
    frame: Any
    unmasked: Any
    intercept_key: Any
    def __init__(self, origin: Any, frame: Any, *, radii: Any = None, gravity: Any = None,
        elevation: float = 0.0, modes: Any = None, epoch: float = 0.0) -> None: ...
    def coords_from_vector3(self, pos: Any, *, obs: Any = None, time: Any = None,
        axes: int = 2, derivs: bool = False, hints: Any = None) -> Any: ...
    def vector3_from_coords(self, coords: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None) -> Any: ...
    def intercept(self, obs: Any, los: Any, *, time: Any = None, direction: str = 'dep',
        derivs: bool = False, guess: Any = None, hints: Any = None) -> Any: ...
    def normal(self, pos: Any, *, obs: Any = None, time: Any = None, derivs: bool = False,
        hints: Any = None) -> Any: ...
    def velocity(self, pos: Any, *, obs: Any = None, time: Any = None) -> Any: ...

##########################################################################################
