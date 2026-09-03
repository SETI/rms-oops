##########################################################################################
# oops/path/fixedpath.pyi
##########################################################################################
"""Type stub for :mod:`oops.path.fixedpath`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.event import Event as Event
from oops.frame.frame_ import Frame as Frame
from oops.path.path_ import Path as Path

class FixedPath(Path):
    def __init__(self, pos: Any, origin: Any, frame: Any = None, *,
        path_id: Any = None) -> None: ...
    def event_at_time(self, time: Any, *, quick: Any = None) -> Any: ...

##########################################################################################
