##########################################################################################
# oops/hosts/juno/sru/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.sru`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.juno import Juno as Juno

def from_file(filespec: Any, return_all_planets: bool = False, method: str = 'strict',
    **parameters: Any) -> Any: ...

class _Metadata:
    nlines: Any
    nsamples: Any
    tstart: Any
    tstop: Any
    exposure: Any
    unit: Any
    tdi_on: Any
    target: Any
    def __init__(self, label: Any) -> None: ...
class SRU:
    SAMPLES: int
    LINES: int
    UV_LOS: Any
    FL_PIXELS: float
    DISTORTION: Any
    spice_frames: Any
    initialized: bool
    @staticmethod
    def initialize(asof: Any = None, **kwargs: Any) -> None: ...
    @staticmethod
    def fov() -> Any: ...
    @staticmethod
    def create_frame(unit: Any, time: Any) -> Any: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
