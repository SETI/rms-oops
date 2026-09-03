##########################################################################################
# oops/fov/offsetfov.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov.offsetfov`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.fittable import Fittable as Fittable
from oops.fov import FOV as FOV

class OffsetFOV(FOV, Fittable):
    fov: Any
    uv_offset: Any
    xy_offset: Any
    uv_shape: Any
    uv_scale: Any
    uv_area: Any
    uv_los: Any
    def __init__(self, fov: Any, uv_offset: Any = None,
        xy_offset: Any = None) -> None: ...
    nparams: int
    @property
    def params(self) -> Any: ...
    def xy_from_uvt(self, uv_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...
    def uv_from_xyt(self, xy_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...

##########################################################################################
