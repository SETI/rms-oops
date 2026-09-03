##########################################################################################
# oops/lightsource.pyi
##########################################################################################
"""Type stub for :mod:`oops.lightsource`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.body import Body as Body
from oops.constants import C as C, RPD as RPD, RPS as RPS
from oops.event import Event as Event
from oops.oops import Oops as Oops
from oops.path import Path as Path

class LightSource(Oops):
    name: Any
    source: Any
    source_is_moving: bool
    shape: Any
    weight: Any
    def __init__(self, name: Any, source: Any, weight: Any = None) -> None: ...
    def photon_to_event(self, event: Any, derivs: bool = False, guess: Any = None,
        antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...
    def as_path(self) -> Any: ...
class DiskSource(LightSource):
    name: Any
    source: Any
    source_is_moving: Any
    xy_grid: Any
    shape: Any
    radius: Any
    weight: Any
    def __init__(self, name: Any, source: Any, radius: Any, size: int = 11,
        compress: bool = False) -> None: ...
    def photon_to_event(self, event: Any, derivs: bool = False, guess: Any = None,
        antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

##########################################################################################
