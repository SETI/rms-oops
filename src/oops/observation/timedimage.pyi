##########################################################################################
# oops/observation/timedimage.pyi
##########################################################################################
"""Type stub for :mod:`oops.observation.timedimage`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.frame import Frame as Frame
from oops.observation import Observation as Observation
from oops.observation.snapshot import Snapshot as Snapshot
from oops.path import Path as Path

class TimedImage(Observation):
    path: Any
    frame: Any
    fov: Any
    u_axis: Any
    v_axis: Any
    t_axis: Any
    swap_uv: Any
    cadence: Any
    shape: Any
    uv_shape: Any
    subfields: Any
    def __init__(self, axes: Any, cadence: Any, fov: Any, path: Any, frame: Any,
        **subfields: Any) -> None: ...
    def uvt(self, indices: Any, *, remask: bool = False, derivs: bool = True) -> Any: ...
    def uvt_range(self, indices: Any, *, remask: bool = False) -> Any: ...
    def time_range_at_uv(self, uv_pair: Any, *, remask: bool = False) -> Any: ...
    def uv_range_at_time(self, time: Any, *, remask: bool = False) -> Any: ...
    def uv_range_at_tstep(self, tstep: Any, *, remask: bool = False) -> Any: ...
    def time_shift(self, dtime: Any) -> Any: ...
    def inventory(self, bodies: Any, **kwargs: Any) -> Any: ...

##########################################################################################
