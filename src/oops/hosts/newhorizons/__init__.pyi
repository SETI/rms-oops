##########################################################################################
# oops/hosts/newhorizons/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.newhorizons`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['lrange', 'NewHorizons']

def lrange(*args: Any) -> Any: ...

class NewHorizons:
    START_TIME: str
    STOP_TIME: str
    initialized: bool
    time: Any
    asof: Any
    meta: Any
    names: Any
    @staticmethod
    def initialize(asof: Any = None, time: Any = None, meta: Any = None) -> Any: ...
    @staticmethod
    def reset() -> None: ...
    @staticmethod
    def spice_instrument_kernel(inst_name: Any, asof: Any = None) -> Any: ...
    @staticmethod
    def spice_frames_kernel(asof: Any = None) -> Any: ...

##########################################################################################
