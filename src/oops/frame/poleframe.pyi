##########################################################################################
# oops/frame/poleframe.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.poleframe`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.cache import Cache as Cache
from oops.frame import Frame as Frame
from oops.transform import Transform as Transform

class PoleFrame(Frame):
    def __init__(self, frame: Any, pole: Any, *, retrograde: bool = False,
        aries: bool = False, frame_id: Any = None, cache_size: int = 100) -> None: ...
    def transform_at_time(self, time: Any, *, quick: Any = None) -> Any: ...
    def node_at_time(self, time: Any, *, quick: Any = None) -> Any: ...

##########################################################################################
