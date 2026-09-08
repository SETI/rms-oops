##########################################################################################
# oops/fov/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from numpy import ndarray
from polymath import Boolean, Matrix, Pair, Scalar, Vector3
# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `polymath.typedefs` names each of those unions.
from polymath.typedefs import PairLike, ScalarLike, Vector3Like
from oops import Fittable as Fittable
from oops.frame import Cmatrix as Cmatrix
from oops.mutable import Mutable as Mutable

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
    def area_factor(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        remask: bool = False, **kwargs: Any) -> Scalar: ...
    def los_from_xy(self, xy_pair: PairLike, *, derivs: bool = False) -> Vector3: ...
    def xy_from_los(self, los: Vector3Like, *, derivs: bool = False) -> Pair: ...
    def los_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Vector3: ...
    def los_from_uv(self, uv_pair: PairLike, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Vector3: ...
    def uv_from_los_t(self, los: Vector3Like, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_los(self, los: Vector3Like, *, derivs: bool = False,
        remask: bool = False, **kwargs: Any) -> Pair: ...
    def offset_angles_from_duv(self, duv: PairLike, *, time: ScalarLike | None = None,
        origin: PairLike | None = None) -> tuple[Scalar, Scalar]: ...
    def offset_duv_from_angles(self, angles: tuple[ScalarLike, ScalarLike] | tuple, *,
        time: ScalarLike | None = None, origin: PairLike | None = None) -> Pair: ...
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
    coefft_xy_from_uv: ndarray | None
    coefft_uv_from_xy: ndarray | None
    dcoefft_xy_from_uv: ndarray
    dcoefft_uv_from_xy: ndarray
    uv_scale: Pair
    uv_shape: Pair
    uv_los: Pair
    iters: int
    fast: bool
    flat_fov: FlatFOV
    uv_area: float
    uv_precision: float
    xy_precision: float
    def __init__(self, uv_scale: PairLike, uv_shape: PairLike, *,
        coefft_xy_from_uv: ndarray | None = None,
        coefft_uv_from_xy: ndarray | None = None, uv_los: PairLike | None = None,
        uv_area: float | None = None, iters: int = 8, fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class FlatFOV(FOV):
    uv_scale: Pair
    uv_shape: Pair
    uv_los: Pair
    uv_area: float
    dxy_duv: Pair
    duv_dxy: Pair
    def __init__(self, uv_scale: PairLike, uv_shape: PairLike, *,
        uv_los: PairLike | None = None, uv_area: float | None = None) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class GapFOV(FOV):
    fov: FOV
    uv_size: Pair
    uv_size_inv: Pair
    uv_scale: Pair
    uv_los: Pair
    uv_area: float
    uv_shape: Pair
    def __init__(self, fov: FOV, uv_size: PairLike) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class NullFOV(FOV):
    uv_los: Pair
    uv_scale: Pair
    uv_shape: Pair
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
    fov: FOV
    uv_offset: Pair
    xy_offset: Pair
    uv_shape: Pair
    uv_scale: Pair
    uv_area: float
    uv_los: Pair
    def __init__(self, fov: FOV, uv_offset: PairLike | None = None,
        xy_offset: PairLike | None = None) -> None: ...
    nparams: int
    @property
    def params(self) -> tuple[float, float]: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class Platescale(FOV, Fittable):
    factor: float
    fov: FOV
    uv_los: Pair
    uv_shape: Pair
    def __init__(self, factor: float, fov: FOV) -> None: ...
    nparams: int
    @property
    def params(self) -> tuple[float]: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class PolynomialFOV(FOV):
    DEBUG: bool
    coefft_xy_from_uv: ndarray | None
    coefft_uv_from_xy: ndarray | None
    coefft_dxy_du: ndarray
    coefft_dxy_dv: ndarray
    coefft_duv_dx: ndarray
    coefft_duv_dy: ndarray
    iters: int
    fast: bool
    uv_shape: Pair
    uv_los: Pair
    flat_fov: FlatFOV
    uv_precision: float
    xy_precision: float
    uv_scale: Pair
    uv_area: float
    def __init__(self, uv_shape: PairLike, coefft_xy_from_uv: ndarray | None = None,
        coefft_uv_from_xy: ndarray | None = None, uv_los: PairLike | None = None,
        uv_area: float | None = None, iters: int = 8, fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class SliceFOV(FOV):
    fov: FOV
    uv_origin: Pair
    uv_shape: Pair
    uv_los: Pair
    uv_scale: Pair
    uv_area: float
    def __init__(self, fov: FOV, origin: PairLike, shape: PairLike) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class Subarray(FOV):
    fov: FOV
    new_los_in_old_uv: Pair
    new_los_wrt_old_xy: Pair
    uv_shape: Pair
    uv_los: Pair
    new_origin_in_old_uv: Pair
    uv_scale: Pair
    uv_area: float
    def __init__(self, fov: FOV, new_los: PairLike, uv_shape: PairLike,
        uv_los: PairLike | None = None) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class SubsampledFOV(FOV):
    fov: FOV
    rescale: Pair
    rescale2: float
    uv_scale: Pair
    uv_los: Pair
    uv_area: float
    uv_shape: Pair
    def __init__(self, fov: FOV, rescale: PairLike) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class TDIFOV(FOV):
    IS_TIME_INDEPENDENT: bool
    fov: FOV
    tstop: float
    tdi_texp: float
    tdi_axis: str
    tdi_sign: int
    uv_los: Pair
    uv_scale: Pair
    uv_shape: Pair
    uv_area: float
    def __init__(self, fov: FOV, tstop: float, tdi_texp: float,
        tdi_axis: str) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...

class WCSFOV(FOV):
    header: dict
    ref_axis: str
    fast: bool
    uv_shape: Pair
    uv_los: Pair
    polyfov: PolynomialFOV | FlatFOV
    cd: Matrix
    clock: float
    rotmat: Matrix
    cdp: Matrix
    cdp_inv: Matrix
    neg_cdp: Matrix
    neg_cdp_inv: Matrix
    uv_scale: Pair
    uv_area: float
    ra: float
    dec: float
    cmatrix: Cmatrix
    def __init__(self, header: dict, ref_axis: str = 'y', fast: bool = True) -> None: ...
    def xy_from_uvt(self, uv_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def uv_from_xyt(self, xy_pair: PairLike, time: ScalarLike | None = None, *,
        derivs: bool = False, remask: bool = False, **kwargs: Any) -> Pair: ...
    def wcs_from_uv(self, uv: PairLike, *, derivs: bool = False,
        remask: bool = False) -> Pair: ...

##########################################################################################
