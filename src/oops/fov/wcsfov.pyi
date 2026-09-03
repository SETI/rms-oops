##########################################################################################
# oops/fov/wcsfov.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov.wcsfov`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.constants import DPR as DPR, RPD as RPD
from oops.fov import FOV as FOV
from oops.fov.flatfov import FlatFOV as FlatFOV
from oops.fov.polynomialfov import PolynomialFOV as PolynomialFOV
from oops.frame.cmatrix import Cmatrix as Cmatrix

class WCSFOV(FOV):
    header: Any
    ref_axis: Any
    fast: Any
    uv_shape: Any
    uv_los: Any
    polyfov: Any
    cd: Any
    clock: Any
    rotmat: Any
    cdp: Any
    cdp_inv: Any
    neg_cdp: Any
    neg_cdp_inv: Any
    uv_scale: Any
    uv_area: Any
    ra: Any
    dec: Any
    cmatrix: Any
    def __init__(self, header: Any, ref_axis: str = 'y', fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...
    def uv_from_xyt(self, xy_pair: Any, time: Any = None, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Any: ...
    def wcs_from_uv(self, uv: Any, *, derivs: bool = False,
        remask: bool = False) -> Any: ...

##########################################################################################
