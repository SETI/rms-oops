##########################################################################################
# oops/hosts/galileo/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.galileo`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from numpy import ndarray
from spicedb import KernelInfo as KernelInfo

__all__ = ['Galileo']

class Galileo:
    START_TIME: str
    STOP_TIME: str
    MONTHS: int
    TDB0: float
    TDB1: float
    DTDB: float
    SLOP: float
    CK_LOADED: ndarray
    CK_LIST: ndarray
    CK_DICT: dict[str, KernelInfo]
    SPK_LOADED: ndarray
    SPK_LIST: ndarray
    SPK_DICT: dict[str, KernelInfo]
    loaded_instruments: list[str]
    initialized: bool
    @staticmethod
    def initialize(planets: list | None = None, asof: str | None = None,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...
    @staticmethod
    def load_kernels() -> None: ...
    @staticmethod
    def initialize_kernels(kernels: list[KernelInfo], lists: ndarray) -> None: ...
    @staticmethod
    def load_instruments(instruments: list[str] = [],
        asof: str | None = None) -> None: ...
    @staticmethod
    def spice_instrument_kernel(inst: str | list | tuple,
        asof: str | None = None) -> tuple[dict, str]: ...
    @staticmethod
    def used_kernels(time: tuple[float, float], inst: str | list | tuple,
        return_all_planets: bool = False) -> list[str]: ...

##########################################################################################
