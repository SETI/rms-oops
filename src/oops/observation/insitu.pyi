##########################################################################################
# oops/observation/insitu.pyi
##########################################################################################
"""Type stub for :mod:`oops.observation.insitu`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.cadence import Cadence as Cadence
from oops.cadence.instant import Instant as Instant
from oops.fov.nullfov import NullFOV as NullFOV
from oops.frame import Frame as Frame
from oops.observation import Observation as Observation
from oops.path import Path as Path

class InSitu(Observation):
    path: Any
    frame: Any
    fov: Any
    cadence: Any
    u_axis: int
    v_axis: int
    swap_uv: bool
    uv_shape: Any
    shape: Any
    t_axis: Any
    subfields: Any
    def __init__(self, cadence: Any, path: Any, **subfields: Any) -> None: ...
    def time_shift(self, dtime: Any) -> Any: ...

##########################################################################################
