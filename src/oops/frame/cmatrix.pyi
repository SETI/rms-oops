##########################################################################################
# oops/frame/cmatrix.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame.cmatrix`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.constants import RPD as RPD
from oops.frame import Frame as Frame
from oops.transform import Transform as Transform

class Cmatrix(Frame):
    def __init__(self, cmatrix: Any, reference: Any = None, *,
        frame_id: Any = None) -> None: ...
    @property
    def transform(self) -> Any: ...
    @staticmethod
    def from_ra_dec(ra: Any, dec: Any, clock: Any, reference: Any = None, *,
        frame_id: Any = None) -> Any: ...
    def transform_at_time(self, time: Any, *, quick: bool = False) -> Any: ...

##########################################################################################
