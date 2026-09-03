##########################################################################################
# oops/surface/centricspheroid.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.centricspheroid`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.surface.centricellipsoid import CentricEllipsoid as CentricEllipsoid
from oops.surface.spheroid import Spheroid as Spheroid

class CentricSpheroid(Spheroid):
    def coords_from_vector3(self, pos: Any, *, obs: Any = None, time: Any = None,
        axes: int = 2, derivs: bool = False, hints: Any = None,
        groundtrack: bool = False) -> Any: ...
    def vector3_from_coords(self, coords: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None, groundtrack: bool = False) -> Any: ...
    def lat_to_centric(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...
    def lat_from_centric(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...
    def lat_to_graphic(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...
    def lat_from_graphic(self, lat: Any, lon: Any = None, *,
        derivs: bool = False) -> Any: ...

##########################################################################################
