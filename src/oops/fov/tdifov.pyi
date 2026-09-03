##########################################################################################
# oops/fov/tdifov.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov.tdifov`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.fov import FOV as FOV

class TDIFOV(FOV):
    IS_TIME_INDEPENDENT: bool
    fov: Any
    tstop: Any
    tdi_texp: Any
    tdi_axis: Any
    tdi_sign: Any
    uv_los: Any
    uv_scale: Any
    uv_shape: Any
    uv_area: Any
    def __init__(self, fov: Any, tstop: Any, tdi_texp: Any, tdi_axis: Any) -> None: ...
    def xy_from_uvt(self, uv_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...
    def uv_from_xyt(self, xy_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...

##########################################################################################
