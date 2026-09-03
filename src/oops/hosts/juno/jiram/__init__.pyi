##########################################################################################
# oops/hosts/juno/jiram/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.juno.jiram`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['from_file', 'JIRAM']

def from_file(filespec: Any, return_all_planets: bool = False, method: str = 'strict',
    **parameters: Any) -> Any: ...

class _Metadata:
    tstart: Any
    tstop: Any
    def __init__(self, label: Any) -> None: ...
class JIRAM:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    @staticmethod
    def initialize(asof: Any = None, **kwargs: Any) -> None: ...
    @staticmethod
    def create_frame(time: Any, name: Any) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
