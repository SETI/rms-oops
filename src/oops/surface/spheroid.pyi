##########################################################################################
# oops/surface/spheroid.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.spheroid`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING, SURFACE_PHOTONS as SURFACE_PHOTONS
from oops.surface.ellipsoid import Ellipsoid as Ellipsoid

class Spheroid(Ellipsoid):
    def __init__(self, origin: Any, frame: Any, radii: Any) -> None: ...
    def intercept_normal_to(self, pos: Any, *, obs: Any = None, time: Any = None,
        direction: str = 'dep', derivs: bool = False, guess: Any = None,
        hints: Any = None) -> Any: ...
    def lon_to_centric(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lon_from_centric(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lon_to_graphic(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lon_from_graphic(self, lon: Any, *, derivs: bool = False) -> Any: ...
    def lat_to_centric(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...
    def lat_from_centric(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...
    def lat_to_graphic(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...
    def lat_from_graphic(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...

##########################################################################################
