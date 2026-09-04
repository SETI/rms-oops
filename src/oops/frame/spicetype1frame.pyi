##########################################################################################
# oops/frame/spicetype1frame.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.spicetype1frame`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops._cache import _Cache as _Cache
from oops.frame import (Frame as Frame, J2000Frame as J2000Frame,
                        LinkedFrame as LinkedFrame)
from oops.frame.spiceframe import SpiceFrame as SpiceFrame
from oops.transform import Transform as Transform

class SpiceType1Frame(SpiceFrame):
    def __init__(self, spice_frame: Any, tick_tolerance: Any, reference: Any = None, *,
        frame_id: Any = None, cache_size: int = 100) -> None: ...
    def transform_at_time(self, time: Any, *, quick: Any = None) -> Any: ...
    def transform_at_time_if_possible(self, time: Any, *, quick: Any = None) -> Any: ...
    @staticmethod
    def get(spice_frame: Any,  # type: ignore[override]
        tick_tolerance: Any, reference: Any = None, *,
        frame_id: Any = None, cache_size: Any = None) -> Any: ...

##########################################################################################
