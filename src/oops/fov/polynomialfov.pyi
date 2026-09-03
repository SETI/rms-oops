##########################################################################################
# oops/fov/polynomialfov.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov.polynomialfov`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING
from oops.fov import FOV as FOV
from oops.fov.flatfov import FlatFOV as FlatFOV

EPSILON: Any

class PolynomialFOV(FOV):
    DEBUG: bool
    coefft_xy_from_uv: Any
    coefft_uv_from_xy: Any
    coefft_dxy_du: Any
    coefft_dxy_dv: Any
    coefft_duv_dx: Any
    coefft_duv_dy: Any
    iters: Any
    fast: Any
    uv_shape: Any
    uv_los: Any
    flat_fov: Any
    uv_precision: Any
    xy_precision: Any
    uv_scale: Any
    uv_area: Any
    def __init__(self, uv_shape: Any, coefft_xy_from_uv: Any = None,
        coefft_uv_from_xy: Any = None, uv_los: Any = None, uv_area: Any = None,
        iters: int = 8, fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...
    def uv_from_xyt(self, xy_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...

##########################################################################################
