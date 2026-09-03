##########################################################################################
# oops/hosts/voyager/iss.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.voyager.iss`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

def from_file(filespec: Any, astrometry: bool = False, action: str = 'error',
    method: str = 'strict', parameters: Any = None) -> Any: ...

def from_index(filespec: Any, geomed: bool = False, action: str = 'ignore',
    omit: bool = True, parameters: Any = {}) -> Any: ...

class ISS:
    fovs: Any
    frames: Any
    initialized: bool
    @staticmethod
    def initialize(asof: Any = None) -> None: ...

##########################################################################################
