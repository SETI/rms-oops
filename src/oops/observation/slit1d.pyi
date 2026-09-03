##########################################################################################
# oops/observation/slit1d.pyi
##########################################################################################
"""Type stub for :mod:`oops.observation.slit1d`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.cadence import Cadence as Cadence
from oops.cadence.snapcadence import SnapCadence as SnapCadence
from oops.frame import Frame as Frame
from oops.observation import Observation as Observation
from oops.path import Path as Path

class Slit1D(Observation):
    path: Any
    frame: Any
    fov: Any
    uv_shape: Any
    shape: Any
    u_axis: Any
    v_axis: int
    swap_uv: bool
    t_axis: int
    cadence: Any
    subfields: Any
    def __init__(self, axes: Any, tstart: Any, texp: Any, fov: Any, path: Any, frame: Any,
        **subfields: Any) -> None: ...
    def uvt(self, indices: Any, *, remask: bool = False, derivs: bool = True) -> Any: ...
    def uvt_range(self, indices: Any, *, remask: bool = False) -> Any: ...
    def time_range_at_uv(self, uv_pair: Any, *, remask: bool = False) -> Any: ...
    def uv_range_at_time(self, time: Any, *, remask: bool = False) -> Any: ...
    def uv_range_at_tstep(self, tstep: Any, *, remask: bool = False) -> Any: ...
    def time_shift(self, dtime: Any) -> Any: ...

##########################################################################################
