##########################################################################################
# oops/observation/snapshot.pyi
##########################################################################################
"""Type stub for :mod:`oops.observation.snapshot`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.body import Body as Body
from oops.cadence import Cadence as Cadence
from oops.cadence.snapcadence import SnapCadence as SnapCadence
from oops.event import Event as Event
from oops.frame import Frame as Frame
from oops.observation import Observation as Observation
from oops.path import Path as Path
from oops.path.multipath import MultiPath as MultiPath

class Snapshot(Observation):
    path: Any
    frame: Any
    fov: Any
    uv_shape: Any
    u_axis: Any
    v_axis: Any
    swap_uv: Any
    t_axis: int
    shape: Any
    cadence: Any
    subfields: Any
    def __init__(self, axes: Any, tstart: Any, texp: Any, fov: Any, path: Any, frame: Any,
        **subfields: Any) -> None: ...
    def uvt(self, indices: Any, *, remask: bool = False, derivs: bool = True) -> Any: ...
    def uvt_range(self, indices: Any, *, remask: bool = False) -> Any: ...
    def uv_range_at_tstep(self, tstep: Any, *, remask: bool = False) -> Any: ...
    def time_range_at_uv(self, uv_pair: Any, *, remask: bool = False) -> Any: ...
    def uv_range_at_time(self, time: Any, *, remask: bool = False) -> Any: ...
    def time_shift(self, dtime: Any) -> Any: ...
    def uv_from_ra_and_dec(self, ra: Any, dec: Any, *, tfrac: float = 0.5,
        time: Any = None, apparent: bool = True, derivs: bool = False, iters: int = 2,
        quick: Any = None) -> Any: ...
    def uv_from_path(self, path: Any, *, tfrac: float = 0.5, time: Any = None,
        derivs: bool = False, guess: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def uv_from_coords(self, surface: Any, coords: Any, *, tfrac: float = 0.5,
        time: Any = None, underside: bool = False, derivs: bool = False,
        quick: Any = None, converge: Any = None) -> Any: ...
    def inventory(self, bodies: Any, *, tfrac: Any = None, time: Any = None,
        expand: float = 0.0, return_type: str = 'list', fov: Any = None,
        quick: Any = None, converge: Any = None) -> Any: ...

##########################################################################################
