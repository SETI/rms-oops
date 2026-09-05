##########################################################################################
# oops/fov/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from numpy import ndarray, number
from polymath import Boolean, Pair, Scalar, Vector3
from oops import Fittable as Fittable
from oops.mutable import Mutable as Mutable

# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `str` is excluded deliberately: no polymath constructor accepts one.
_Numeric = float | number | list['_Numeric'] | tuple['_Numeric', ...]
PairLike = Pair | ndarray | _Numeric
ScalarLike = Scalar | ndarray | _Numeric
Vector3Like = Vector3 | ndarray | _Numeric

__all__ = ['FOV', 'BarrelFOV', 'FlatFOV', 'GapFOV', 'NullFOV', 'OffsetFOV', 'Platescale',
           'PolynomialFOV', 'SliceFOV', 'Subarray', 'SubsampledFOV', 'TDIFOV', 'WCSFOV']

class FOV(Mutable):
    IS_TIME_INDEPENDENT: bool
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def xy_from_uv(self, uv_pair: PairLike, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xy(self, xy_pair: PairLike, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Pair: ...
    def area_factor(self, uv_pair: PairLike | ndarray | tuple,
        time: ScalarLike | None = None, *, remask: bool = False,
        **kwargs: Any) -> Scalar: ...
    def los_from_xy(self, xy_pair: PairLike, *, derivs: bool = False) -> Vector3: ...
    def xy_from_los(self, los: Vector3Like, *, derivs: bool = False) -> Pair: ...
    def los_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Vector3: ...
    def los_from_uv(self, uv_pair: PairLike | ndarray | tuple, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Vector3: ...
    def uv_from_los_t(self, los: Vector3Like, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_los(self, los: Vector3Like, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Pair: ...
    def offset_angles_from_duv(self, duv: PairLike, *, time: ScalarLike | None = None,
        origin: PairLike | None = None) -> tuple[Scalar, Scalar]: ...
    def offset_duv_from_angles(self, angles: tuple, *, time: ScalarLike | None = None,
        origin: PairLike | None = None) -> Pair: ...
    def uv_is_outside(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        uv_min: PairLike | None = None, uv_max: PairLike | None = None,
        inclusive: bool = True) -> Boolean: ...
    def u_or_v_is_outside(self, uv_pair: PairLike, uv_index: int, *,
        uv_min: PairLike | None = None, uv_max: PairLike | None = None,
        inclusive: bool = True) -> Boolean: ...
    def xy_is_outside(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        inclusive: bool = True, uv_min: PairLike | None = None,
        uv_max: PairLike | None = None, **kwargs: Any) -> Boolean: ...
    def los_is_outside(self, los: Vector3Like, time: ScalarLike | None = None, *,
        inclusive: bool = True, uv_min: PairLike | None = None,
        uv_max: PairLike | None = None, **kwargs: Any) -> Boolean: ...
    def nearest_uv(self, uv_pair: PairLike | tuple | list | ndarray, *,
        remask: bool = False) -> Pair: ...
    def max_inversion_error(self, steps: int = 30) -> float: ...
    def center_xy(self, time: ScalarLike | None = None) -> Pair: ...
    def center_los(self, time: ScalarLike | None = None) -> Vector3: ...
    @property
    def center_dlos_duv(self) -> Vector3: ...
    @property
    def outer_radius(self) -> float: ...
    @property
    def inner_radius(self) -> float: ...
    def corner00_xy(self, time: ScalarLike | None = None) -> Pair: ...
    def corner01_xy(self, time: ScalarLike | None = None) -> Pair: ...
    def corner10_xy(self, time: ScalarLike | None = None) -> Pair: ...
    def corner11_xy(self, time: ScalarLike | None = None) -> Pair: ...
    def sphere_falls_inside(self, center: Vector3Like, radius: ScalarLike, *,
        time: ScalarLike | None = None, border: float = 0.0) -> Boolean: ...

class BarrelFOV(FOV):
    DEBUG: bool
    coefft_xy_from_uv: Any
    coefft_uv_from_xy: Any
    dcoefft_xy_from_uv: Any
    dcoefft_uv_from_xy: Any
    uv_scale: Any
    uv_shape: Any
    uv_los: Any
    iters: Any
    fast: Any
    flat_fov: Any
    uv_area: Any
    uv_precision: Any
    xy_precision: Any
    def __init__(self, uv_scale: PairLike | tuple, uv_shape: PairLike | tuple, *,
        coefft_xy_from_uv: ndarray | None = None,
        coefft_uv_from_xy: ndarray | None = None, uv_los: PairLike | tuple | None = None,
        uv_area: float | None = None, iters: int = 8, fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class FlatFOV(FOV):
    uv_scale: Any
    uv_shape: Any
    uv_los: Any
    uv_area: Any
    dxy_duv: Any
    duv_dxy: Any
    def __init__(self, uv_scale: PairLike | float | tuple[float, float],
        uv_shape: PairLike | int | tuple[int, int], *,
        uv_los: PairLike | tuple | None = None, uv_area: float | None = None) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class GapFOV(FOV):
    fov: Any
    uv_size: Any
    uv_size_inv: Any
    uv_scale: Any
    uv_los: Any
    uv_area: Any
    uv_shape: Any
    def __init__(self, fov: FOV, uv_size: float | tuple | PairLike) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class NullFOV(FOV):
    uv_los: Any
    uv_scale: Any
    uv_shape: Any
    uv_area: float
    def __init__(self) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def area_factor(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        remask: bool = False, **kwargs: Any) -> Scalar: ...
    def los_from_xy(self, xy_pair: PairLike, *, derivs: bool = False) -> Vector3: ...
    def xy_from_los(self, los: Vector3Like, *, derivs: bool = False) -> Pair: ...
    def los_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Vector3: ...
    def uv_from_los_t(self, los: Vector3Like, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_is_outside(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        uv_min: PairLike | None = None, uv_max: PairLike | None = None,
        inclusive: bool = True) -> Boolean: ...
    def u_or_v_is_outside(self, uv_pair: PairLike, uv_index: int, *,
        uv_min: PairLike | None = None, uv_max: PairLike | None = None,
        inclusive: bool = True) -> Boolean: ...
    def xy_is_outside(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        inclusive: bool = True, uv_min: PairLike | None = None,
        uv_max: PairLike | None = None, **kwargs: Any) -> Boolean: ...
    def los_is_outside(self, los: Vector3Like, time: ScalarLike | None = None, *,
        inclusive: bool = True, uv_min: PairLike | None = None,
        uv_max: PairLike | None = None, **kwargs: Any) -> Boolean: ...
    def nearest_uv(self, uv_pair: PairLike, *, remask: bool = False) -> Pair: ...
    def max_inversion_error(self, steps: int = 30) -> float: ...

class OffsetFOV(FOV, Fittable):
    fov: Any
    uv_offset: Any
    xy_offset: Any
    uv_shape: Any
    uv_scale: Any
    uv_area: Any
    uv_los: Any
    def __init__(self, fov: FOV, uv_offset: PairLike | None = None,
        xy_offset: PairLike | None = None) -> None: ...
    nparams: int
    @property
    def params(self) -> Any: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class Platescale(FOV, Fittable):
    factor: Any
    fov: Any
    uv_los: Any
    uv_shape: Any
    def __init__(self, factor: float, fov: FOV) -> None: ...
    nparams: int
    @property
    def params(self) -> Any: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

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
    def __init__(self, uv_shape: float | tuple | PairLike,
        coefft_xy_from_uv: ndarray | None = None,
        coefft_uv_from_xy: ndarray | None = None,
        uv_los: float | tuple | PairLike | None = None, uv_area: float | None = None,
        iters: int = 8, fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class SliceFOV(FOV):
    fov: Any
    uv_origin: Any
    uv_shape: Any
    uv_los: Any
    uv_scale: Any
    uv_area: Any
    def __init__(self, fov: FOV, origin: tuple | PairLike,
        shape: float | tuple | PairLike) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class Subarray(FOV):
    fov: Any
    new_los_in_old_uv: Any
    new_los_wrt_old_xy: Any
    uv_shape: Any
    uv_los: Any
    new_origin_in_old_uv: Any
    uv_scale: Any
    uv_area: Any
    def __init__(self, fov: FOV, new_los: tuple | PairLike,
        uv_shape: float | tuple | PairLike,
        uv_los: float | tuple | PairLike | None = None) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class SubsampledFOV(FOV):
    fov: Any
    rescale: Any
    rescale2: Any
    uv_scale: Any
    uv_los: Any
    uv_area: Any
    uv_shape: Any
    def __init__(self, fov: FOV, rescale: float | tuple | PairLike) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

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
    def __init__(self, fov: FOV, tstop: float, tdi_texp: float,
        tdi_axis: str) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

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
    def __init__(self, header: dict, ref_axis: str = 'y', fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def wcs_from_uv(self, uv: PairLike, *, derivs: bool = False,
        remask: bool = False) -> Pair: ...

##########################################################################################
