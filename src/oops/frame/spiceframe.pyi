##########################################################################################
# oops/frame/spiceframe.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.spiceframe`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.frame import (Frame as Frame, J2000Frame as J2000Frame,
                        LinkedFrame as LinkedFrame, NullFrame as NullFrame)
from oops.frame.quickframe import QuickFrame as QuickFrame
from oops.transform import Transform as Transform

class SpiceFrame(Frame):
    def __init__(self, spice_frame: Any, reference: Any = None, *,
        omega_type: str = 'tabulated', omega_dt: float = 1.0,
        frame_id: Any = None) -> None: ...
    def transform_at_time(self, time: Any, *, quick: Any = None) -> Any: ...
    def transform_at_time_if_possible(self, time: Any, *, quick: Any = None) -> Any: ...
    @staticmethod
    def get(spice_frame: Any, reference: Any = None, *, omega_type: str = 'tabulated',
        omega_dt: float = 1.0, frame_id: Any = None) -> Any: ...

##########################################################################################
