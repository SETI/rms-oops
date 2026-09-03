##########################################################################################
# oops/frame/navigation.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.navigation`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.fittable import Fittable as Fittable
from oops.frame import Frame as Frame
from oops.transform import Transform as Transform

class Navigation(Frame, Fittable):
    nparams: Any
    def __init__(self, arg: Any, /, reference: Any, *, freeze: bool = False,
        frame_id: Any = None, _matrix: Any = None) -> None: ...
    @property
    def angles(self) -> Any: ...
    @property
    def link(self) -> Any: ...
    @property
    def params(self) -> Any: ...
    def transform_at_time(self, time: Any, *, quick: bool = False) -> Any: ...

##########################################################################################
