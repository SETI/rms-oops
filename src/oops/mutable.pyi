##########################################################################################
# oops/mutable.pyi
##########################################################################################
"""Type stub for :mod:`oops.mutable`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.fittable import Fittable as Fittable
from oops.oops import Oops as Oops
from typing import Any, NamedTuple

class _Info(NamedTuple):
    is_fittable: Any
    is_mutable: Any
    is_frozen: Any
    mutable_names: Any
    unfrozen_names: Any
    versions: Any
def refresh(obj: Any, /) -> bool: ...

def needs_refresh(obj: Any, /) -> bool: ...

def freeze(obj: Any, /) -> bool: ...

def set_param_order(obj: Any, names: list[str]) -> None: ...

def get_param_order(obj: Any) -> list[str]: ...

def get_nparams(obj: Any) -> int: ...

def set_params(obj: Any, params: Any) -> bool: ...

def get_params(obj: Any) -> tuple[float, ...]: ...

def is_fittable(obj: Any, /) -> bool: ...

def is_mutable(obj: Any, /) -> bool: ...

def is_frozen(obj: Any, /) -> bool: ...

def mutable_names(obj: Any, /) -> list[str]: ...

def unfrozen_names(obj: Any, /) -> list[str]: ...

def version(obj: Any, /) -> int: ...

class Mutable(Oops):
    def refresh(self) -> bool: ...
    def freeze(self) -> bool: ...

##########################################################################################
