##########################################################################################
# oops/fittable.pyi
##########################################################################################
"""Type stub for :mod:`oops.fittable`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.oops import Oops as Oops
from typing import Any

class Fittable(Oops):
    @property
    def is_frozen(self) -> bool: ...
    @property
    def version(self) -> int: ...
    def set_params(self, params: Any) -> bool: ...
    def refresh(self) -> bool: ...
    def copy(self) -> Fittable: ...
    def freeze(self) -> bool: ...

##########################################################################################
