##########################################################################################
# oops/frame/frameshift.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.frameshift`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.fittable import Fittable as Fittable
from oops.frame import Frame as Frame

class FrameShift(Frame, Fittable):
    def __init__(self, arg: Any, /, frame: Any, *, frame_id: Any = None,
        freeze: bool = False) -> None: ...
    @property
    def dt(self) -> Any: ...
    @property
    def link(self) -> Any: ...
    nparams: int
    @property
    def params(self) -> Any: ...
    def transform_at_time(self, time: Any, *, quick: Any = None) -> Any: ...

##########################################################################################
