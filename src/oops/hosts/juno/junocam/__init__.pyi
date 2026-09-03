##########################################################################################
# oops/hosts/juno/junocam/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.junocam`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['from_file', 'JUNOCAM']

def from_file(filespec: Any, fast_distortion: bool = True,
    return_all_planets: bool = False, snap: bool = False, method: str = 'strict',
    **parameters: Any) -> Any: ...

class _Metadata:
    nlines: Any
    nsamples: Any
    frlines: int
    nframelets: Any
    exposure: Any
    filter: Any
    tinter: Any
    tinter0: Any
    tstart: Any
    tstart0: Any
    tstop: Any
    tdi_stages: Any
    tdi_texp: Any
    target: Any
    delta: Any
    bias: Any
    fov: Any
    def __init__(self, label: Any) -> None: ...
    def update_cy(self, label: Any, cy: Any) -> Any: ...
class JUNOCAM:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    @staticmethod
    def initialize(asof: Any = None, **kwargs: Any) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
