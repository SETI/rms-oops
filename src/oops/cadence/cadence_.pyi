##########################################################################################
# oops/cadence/cadence_.pyi
##########################################################################################
"""Type stub for :mod:`oops.cadence.cadence_`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.mutable import Mutable as Mutable

class Cadence(Mutable):
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
    def time_is_inside(self, time: Any, *, inclusive: bool = True) -> Any: ...
    def tstride_at_tstep(self, tstep: Any, sign: int = 1, *,
        remask: bool = False) -> Any: ...

##########################################################################################
