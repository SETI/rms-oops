##########################################################################################
# oops/path/linearcoordpath.pyi
##########################################################################################
"""Type stub for :mod:`oops.path.linearcoordpath`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.event import Event as Event
from oops.path.path_ import Path as Path

class LinearCoordPath(Path):
    def __init__(self, surface: Any, coords: Any, coords_dot: Any, epoch: Any, *,
        obs: Any = None, path_id: Any = None) -> None: ...
    def event_at_time(self, time: Any, *, quick: Any = None) -> Any: ...

##########################################################################################
