##########################################################################################
# oops/cadence/sequence.pyi
##########################################################################################
"""Type stub for :mod:`oops.cadence.sequence`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.cadence import Cadence as Cadence

class Sequence(Cadence):
    min_tstride: Any
    max_tstride: Any
    is_continuous: Any
    is_unique: Any
    lasttime: Any
    time: Any
    midtime: Any
    shape: Any
    def __init__(self, tlist: Any, texp: Any) -> None: ...
    def time_at_tstep(self, tstep: Any, *, remask: bool = False, derivs: bool = False,
        inclusive: bool = True) -> Any: ...
    def time_range_at_tstep(self, tstep: Any, *, remask: bool = False,
        inclusive: bool = True, shift: bool = True) -> Any: ...
    def tstep_at_time(self, time: Any, *, remask: bool = False, derivs: bool = False,
        inclusive: bool = True) -> Any: ...
    def tstep_range_at_time(self, time: Any, *, remask: bool = False,
        inclusive: bool = True) -> Any: ...
    def time_is_outside(self, time: Any, *, inclusive: bool = True) -> Any: ...
    def time_shift(self, secs: Any) -> Any: ...
    def as_continuous(self) -> Any: ...

##########################################################################################
