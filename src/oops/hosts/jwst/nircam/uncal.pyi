##########################################################################################
# oops/hosts/jwst/nircam/uncal.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.jwst.nircam.uncal`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.jwst import MASK_VALUES as MASK_VALUES
from oops.hosts.jwst.nircam import NIRCam as NIRCam

ARCSEC_PER_RADIAN: Any
RAW_SATURATION: int
DEBUG: bool

def from_file(filespec: Any, **options: Any) -> Any: ...

class Uncal(NIRCam):
    @staticmethod
    def from_hdulist(hdulist: Any, **options: Any) -> Any: ...
    @staticmethod
    def fit_to_calibrated(raw_hdulist: Any, cal_hdulist: Any, diff_texp: Any,
        cal_factor: float = 0.0) -> Any: ...

##########################################################################################
