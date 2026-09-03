##########################################################################################
# oops/frame/spinframe.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.spinframe`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.frame import Frame as Frame
from oops.transform import Transform as Transform

class SpinFrame(Frame):
    def __init__(self, offset: Any, rate: Any, epoch: Any, axis: Any, reference: Any, *,
        frame_id: Any = None) -> None: ...
    def transform_at_time(self, time: Any, *, quick: bool = False) -> Any: ...

##########################################################################################
