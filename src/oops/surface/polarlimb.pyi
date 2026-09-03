##########################################################################################
# oops/surface/polarlimb.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.polarlimb`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.constants import TWOPI as TWOPI
from oops.surface.limb import Limb as Limb

class PolarLimb(Limb):
    COORDINATE_TYPE: str
    COORDINATE_NAMES: Any
    COORDINATE_ABBREVS: Any
    COORDINATE_RANGES: Any
    def coords_from_vector3(self, pos: Any, *, obs: Any = None, time: Any = None,
        axes: int = 2, derivs: bool = False, hints: Any = None,
        groundtrack: bool = False) -> Any: ...
    def vector3_from_coords(self, coords: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None, groundtrack: bool = False) -> Any: ...

##########################################################################################
