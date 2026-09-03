##########################################################################################
# oops/hosts/newhorizons/lorri.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.newhorizons.lorri`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from . import NewHorizons as NewHorizons

SPICE_TO_FRAME: Any

def radec_from_uv(u: Any, v: Any, header: Any) -> Any: ...

def uv_from_radec(ra: Any, dec: Any, header: Any) -> Any: ...

def to_xms(x: Any) -> Any: ...

def from_file(filespec: Any, geom: str = 'spice', pointing: str = 'spice',
    fov_type: str = 'fast', asof: Any = None, meta: Any = None,
    **parameters: Any) -> Any: ...

def from_index(filespec: Any, fov_type: str = 'fast', asof: Any = None, meta: Any = None,
    **parameters: Any) -> Any: ...

class LORRI:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    asof: Any
    meta: Any
    LORRI_F: float
    LORRI_E2: float
    LORRI_E5: float
    LORRI_E6: float
    LORRI_KX: float
    LORRI_KY: float
    LORRI_COEFF: Any
    LORRI_INV_COEFF: Any
    @staticmethod
    def initialize(asof: Any = None, time: Any = None, meta: Any = None) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
