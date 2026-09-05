##########################################################################################
# oops/hosts/galileo/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.galileo`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['Galileo']

class Galileo:
    START_TIME: str
    STOP_TIME: str
    MONTHS: int
    TDB0: Any
    TDB1: Any
    DTDB: Any
    SLOP: float
    CK_LOADED: Any
    CK_LIST: Any
    CK_DICT: Any
    SPK_LOADED: Any
    SPK_LIST: Any
    SPK_DICT: Any
    loaded_instruments: Any
    initialized: bool
    @staticmethod
    def initialize(planets: list | None = None, asof: str | None = None,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...
    @staticmethod
    def load_kernels() -> None: ...
    @staticmethod
    def initialize_kernels(kernels: Any, lists: Any) -> None: ...
    @staticmethod
    def load_instruments(instruments: Any = [], asof: str | None = None) -> None: ...
    @staticmethod
    def spice_instrument_kernel(inst: str | list | tuple,
        asof: str | None = None) -> tuple: ...
    @staticmethod
    def used_kernels(time: Any, inst: Any, return_all_planets: bool = False) -> Any: ...

##########################################################################################
