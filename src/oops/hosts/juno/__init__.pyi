##########################################################################################
# oops/hosts/juno/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['Juno']

class Juno:
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
    def initialize(ck: str = 'reconstructed', spk: str = 'reconstructed',
        gapfill: bool = True, **kwargs: Any) -> None: ...
    @staticmethod
    def reset() -> None: ...
    @staticmethod
    def load_ck(t: Any) -> None: ...
    @staticmethod
    def load_cks(t0: Any, t1: Any) -> None: ...
    @staticmethod
    def load_spk(t: Any) -> None: ...
    @staticmethod
    def load_spks(t0: Any, t1: Any) -> None: ...
    @staticmethod
    def load_kernels(t0: Any, t1: Any, loaded: Any, lists: Any,
        kernel_dict: Any) -> None: ...
    @staticmethod
    def initialize_kernels(kernels: Any, lists: Any) -> None: ...
    @staticmethod
    def load_instruments(instruments: Any = [], asof: Any = None) -> None: ...
    @staticmethod
    def spice_instrument_kernel(inst: Any, asof: Any = None) -> Any: ...
    @staticmethod
    def spice_frames_kernel(asof: Any = None) -> Any: ...
    @staticmethod
    def used_kernels(time: Any, inst: Any, return_all_planets: bool = False) -> Any: ...

##########################################################################################
