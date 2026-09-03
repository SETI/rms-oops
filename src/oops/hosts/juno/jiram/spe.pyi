##########################################################################################
# oops/hosts/juno/jiram/spe.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.jiram.spe`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.juno.jiram import JIRAM as JIRAM

def from_file(filespec: Any, label: Any, fast_distortion: bool = True,
    return_all_planets: bool = False, **parameters: Any) -> Any: ...

class _Metadata:
    nlines: Any
    nsamples: Any
    exposure: Any
    tstart: Any
    tstop: Any
    target: Any
    fov: Any
    def __init__(self, label: Any) -> None: ...
class SPE:
    initialized: bool
    @staticmethod
    def initialize(time: Any, asof: Any = None, **kwargs: Any) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
