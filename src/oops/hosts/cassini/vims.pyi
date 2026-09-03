##########################################################################################
# oops/hosts/cassini/vims.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.cassini.vims`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.cassini import Cassini as Cassini

TIME_FACTOR: float
IR_NORMAL_PIXEL: float
VIS_NORMAL_PIXEL: float
IR_HIRES_FACTOR: float
VIS_HIRES_FACTOR: float
IR_NORMAL_SCALE: Any
VIS_NORMAL_SCALE: Any
IR_HIRES_SCALE: Any
VIS_HIRES_SCALE: Any
IR_OVER_VIS: Any
IR_FULL_FOV: Any
VIS_FULL_FOV: Any
CORE_DESCRIPTION_FMT: Any
SUFFIX_DESCRIPTION_FMT: Any
BAND_BIN_CENTER_FMT: Any

def from_file(filespec: Any, data: bool = True, method: str = 'strict') -> Any: ...

def meshgrid_and_times(obs: Any, oversample: int = 6, extend: float = 1.5) -> Any: ...

def initialize(ck: str = 'reconstructed', planets: Any = None, asof: Any = None,
    spk: str = 'reconstructed', gapfill: bool = True, mst_pck: bool = True,
    irregulars: bool = True) -> None: ...

class VIMS:
    initialized: bool
    instrument_kernel: Any
    @staticmethod
    def initialize(ck: str = 'reconstructed', planets: Any = None, asof: Any = None,
        spk: str = 'reconstructed', gapfill: bool = True, mst_pck: bool = True,
        irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
