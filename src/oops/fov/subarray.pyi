##########################################################################################
# oops/fov/subarray.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov.subarray`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.fov import FOV as FOV

class Subarray(FOV):
    fov: Any
    new_los_in_old_uv: Any
    new_los_wrt_old_xy: Any
    uv_shape: Any
    uv_los: Any
    new_origin_in_old_uv: Any
    uv_scale: Any
    uv_area: Any
    def __init__(self, fov: Any, new_los: Any, uv_shape: Any,
        uv_los: Any = None) -> None: ...
    def xy_from_uvt(self, uv_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...
    def uv_from_xyt(self, xy_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...

##########################################################################################
