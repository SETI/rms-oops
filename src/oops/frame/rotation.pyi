##########################################################################################
# oops/frame/rotation.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.rotation`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.fittable import Fittable as Fittable
from oops.frame import Frame as Frame
from oops.transform import Transform as Transform

class Rotation(Frame, Fittable):
    def __init__(self, arg: Any, /, axis: Any, reference: Any, *, freeze: bool = False,
        frame_id: Any = None) -> None: ...
    @property
    def angle(self) -> Any: ...
    nparams: int
    @property
    def params(self) -> Any: ...
    def transform_at_time(self, time: Any, *, quick: bool = False) -> Any: ...

##########################################################################################
