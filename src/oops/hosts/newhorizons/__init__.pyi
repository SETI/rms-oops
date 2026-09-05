##########################################################################################
# oops/hosts/newhorizons/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.newhorizons`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
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
    def initialize(asof: str | None = None, time: list | None = None,
        meta: str | None = None) -> list[str]: ...
    @staticmethod
    def reset() -> None: ...
    @staticmethod
    def spice_instrument_kernel(inst_name: Any, asof: str | None = None) -> tuple: ...
    @staticmethod
    def spice_frames_kernel(asof: str | None = None) -> tuple: ...

##########################################################################################
