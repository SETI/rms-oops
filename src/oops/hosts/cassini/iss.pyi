##########################################################################################
# oops/hosts/cassini/iss.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.cassini.iss`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.cassini import Cassini as Cassini

CMATRIX_ROTATION: Any

def from_file(filespec: Any, *, fast_distortion: bool = True,
    return_all_planets: bool = False, frame: Any = None, navigation: bool = False,
    **kwargs: Any) -> Any: ...

def from_index(filespec: Any, fast_distortion: bool = True,
    return_all_planets: bool = False, navigation: bool = False, **kwargs: Any) -> Any: ...

def initialize(ck: str = 'reconstructed', planets: Any = None, asof: Any = None,
    spk: str = 'reconstructed', gapfill: bool = True, mst_pck: bool = True,
    irregulars: bool = True) -> None: ...

class ISS:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    NAC_F: float
    NAC_E2: float
    NAC_E5: float
    NAC_E6: float
    NAC_KX: float
    NAC_KY: float
    NAC_COEFF: Any
    WAC_F: float
    WAC_E2: float
    WAC_E5: float
    WAC_E6: float
    WAC_KX: float
    WAC_KY: float
    WAC_COEFF: Any
    DISTORTION_COEFF_XY_TO_UV: Any
    NAC_INV_COEFF: Any
    WAC_INV_COEFF: Any
    DISTORTION_COEFF_UV_TO_XY: Any
    @staticmethod
    def initialize(ck: str = 'reconstructed', planets: Any = None, asof: Any = None,
        spk: str = 'reconstructed', gapfill: bool = True, mst_pck: bool = True,
        irregulars: bool = True) -> None: ...
    @staticmethod
    def define_camera_frames() -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
