##########################################################################################
# oops/surface/ellipsoid.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.ellipsoid`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING, SURFACE_PHOTONS as SURFACE_PHOTONS
from oops.constants import HALFPI as HALFPI, TWOPI as TWOPI
from oops.frame.frame_ import Frame as Frame
from oops.path.path_ import Path as Path
from oops.surface.surface_ import Surface as Surface

class Ellipsoid(Surface):
    COORDINATE_TYPE: str
    COORDINATE_NAMES: Any
    COORDINATE_ABBREVS: Any
    COORDINATE_RANGES: Any
    IS_VIRTUAL: bool
    HAS_INTERIOR: bool
    origin: Any
    frame: Any
    unmasked: Any
    intercept_key: Any
    def __init__(self, origin: Any, frame: Any, radii: Any) -> None: ...
    @property
    def radii(self) -> Any: ...
    @property
    def unsquash_sq(self) -> Any: ...
    def coords_from_vector3(self, pos: Any, *, obs: Any = None, time: Any = None,
        axes: int = 2, derivs: bool = False, hints: Any = None,
        groundtrack: bool = False) -> Any: ...
    def vector3_from_coords(self, coords: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None, groundtrack: bool = False) -> Any: ...
    def position_is_inside(self, pos: Any, *, obs: Any = None,
        time: Any = None) -> Any: ...
    def intercept(self, obs: Any, los: Any, *, time: Any = None, direction: str = 'dep',
        derivs: bool = False, guess: Any = None, hints: Any = None) -> Any: ...
    def normal(self, pos: Any, *, obs: Any = None, time: Any = None, derivs: bool = False,
        hints: Any = None) -> Any: ...
    def intercept_with_normal(self, normal: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None) -> Any: ...
    def intercept_normal_to(self, pos: Any, *, obs: Any = None, time: Any = None,
        direction: str = 'dep', derivs: bool = False, guess: Any = None,
        hints: Any = None) -> Any: ...
    def lon_to_centric(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lon_from_centric(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lon_to_graphic(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lon_from_graphic(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lat_to_centric(self, lat: Any, lon: Any, *, derivs: bool = False) -> Any: ...
    def lat_from_centric(self, lat: Any, lon: Any, *, derivs: bool = False) -> Any: ...
    def lat_to_graphic(self, lat: Any, lon: Any, *, derivs: bool = False) -> Any: ...
    def lat_from_graphic(self, lat: Any, lon: Any, *, derivs: bool = False) -> Any: ...

##########################################################################################
