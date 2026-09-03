##########################################################################################
# oops/frame/inclinedframe.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.inclinedframe`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.frame import Frame as Frame
from oops.frame.rotation import Rotation as Rotation
from oops.frame.spinframe import SpinFrame as SpinFrame

class InclinedFrame(Frame):
    def __init__(self, inc: Any, node: Any, rate: Any, epoch: Any, *, despin: bool = True,
        reference: Any = None, frame_id: Any = None) -> None: ...
    def transform_at_time(self, time: Any, *, quick: bool = False) -> Any: ...
    def node_at_time(self, time: Any, *, quick: bool = False) -> Any: ...

##########################################################################################
