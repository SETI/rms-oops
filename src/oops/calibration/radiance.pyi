##########################################################################################
# oops/calibration/radiance.pyi
##########################################################################################
"""Type stub for :mod:`oops.calibration.radiance`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.calibration.flatcalib import FlatCalib as FlatCalib

class Radiance(FlatCalib):
    name: Any
    fov: Any
    has_baseline: Any
    shape: Any
    def __init__(self, name: Any, fov: Any, factor: Any,
        baseline: float = 0.0) -> None: ...
    def point_from_dn(self, dn: Any, uv_pair: Any) -> Any: ...
    def dn_from_point(self, value: Any, uv_pair: Any) -> Any: ...
    def prescale(self, factor: Any, baseline: float = 0.0, *, name: str = '') -> Any: ...

##########################################################################################
