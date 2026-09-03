##########################################################################################
# oops/surface/orbitplane.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.orbitplane`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.constants import PI as PI, TWOPI as TWOPI
from oops.frame.frame_ import Frame as Frame
from oops.frame.inclinedframe import InclinedFrame as InclinedFrame
from oops.frame.spinframe import SpinFrame as SpinFrame
from oops.path.circlepath import CirclePath as CirclePath
from oops.path.path_ import Path as Path
from oops.surface.ringplane import RingPlane as RingPlane
from oops.surface.surface_ import Surface as Surface

class OrbitPlane(Surface):
    COORDINATE_TYPE: str
    COORDINATE_NAMES: Any
    COORDINATE_ABBREVS: Any
    COORDINATE_RANGES: Any
    IS_VIRTUAL: bool
    origin: Any
    frame: Any
    intercept_key: Any
    unmasked: Any
    def __init__(self, elements: Any, epoch: Any, origin: Any, frame: Any, *,
        path_id: Any = None, radii: Any = None) -> None: ...
    def coords_from_vector3(self, pos: Any, *, obs: Any = None, time: Any = None,
        axes: int = 2, derivs: bool = False, hints: Any = None) -> Any: ...
    def vector3_from_coords(self, coords: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None) -> Any: ...
    def intercept(self, obs: Any, los: Any, *, time: Any = None, direction: str = 'dep',
        derivs: bool = False, guess: Any = None, hints: Any = None) -> Any: ...
    def normal(self, pos: Any, *, obs: Any = None, time: Any = None, derivs: bool = False,
        hints: Any = None) -> Any: ...
    def velocity(self, pos: Any, *, obs: Any = None, time: Any = None) -> Any: ...
    def from_mean_anomaly(self, anom: Any) -> Any: ...
    def to_mean_anomaly(self, lon: Any) -> Any: ...

##########################################################################################
