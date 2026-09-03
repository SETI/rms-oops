##########################################################################################
# oops/frame/quickframe.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.quickframe`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING, QUICK as QUICK
from oops.frame import Frame as Frame
from oops.transform import Transform as Transform

class QuickFrame(Frame):
    def __init__(self, frame: Any, tmin: Any, tmax: Any, quick: Any = None) -> None: ...
    def transform_at_time(self, time: Any, *, quick: bool = False) -> Any: ...
    def extend(self, tmin: Any, tmax: Any) -> None: ...
    @staticmethod
    def for_frame(frame: Any, time: Any, *, quick: Any = None) -> Any: ...

##########################################################################################
