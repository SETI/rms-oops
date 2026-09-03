##########################################################################################
# oops/path/quickpath.pyi
##########################################################################################
"""Type stub for :mod:`oops.path.quickpath`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING, QUICK as QUICK
from oops.event import Event as Event
from oops.path.path_ import Path as Path

class QuickPath(Path):
    def __init__(self, path: Any, tmin: Any, tmax: Any, quickdict: Any) -> None: ...
    def event_at_time(self, time: Any, *, quick: bool = False) -> Any: ...
    def extend(self, tmin: Any, tmax: Any) -> None: ...
    @staticmethod
    def for_path(path: Any, time: Any, *, quick: Any = None) -> Any: ...

##########################################################################################
