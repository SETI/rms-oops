##########################################################################################
# oops/path/spicepath.pyi
##########################################################################################
"""Type stub for :mod:`oops.path.spicepath`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.event import Event as Event
from oops.frame.frame_ import Frame as Frame, J2000Frame as J2000Frame
from oops.frame.spiceframe import SpiceFrame as SpiceFrame
from oops.path.path_ import LinkedPath as LinkedPath, Path as Path, SSBPath as SSBPath
from oops.path.quickpath import QuickPath as QuickPath

class SpicePath(Path):
    def __init__(self, spice_path: Any, origin: Any = None, frame: Any = None, *,
        path_id: Any = None) -> None: ...
    def event_at_time(self, time: Any, *, quick: Any = None) -> Any: ...
    @staticmethod
    def get(spice_path: Any, origin: Any = None, frame: Any = None, *,
        path_id: Any = None) -> Any: ...

##########################################################################################
