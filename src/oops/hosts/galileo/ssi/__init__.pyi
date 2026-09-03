##########################################################################################
# oops/hosts/galileo/ssi/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.galileo.ssi`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['from_file', 'from_index', 'initialize', 'Metadata', 'SSI']

def from_file(filespec: Any, return_all_planets: bool = False, full_fov: bool = False,
    method: str = 'strict', **parameters: Any) -> Any: ...

def from_index(filespec: Any, supplemental_filespec: Any = None, full_fov: bool = False,
    **parameters: Any) -> Any: ...

def initialize(planets: Any = None, asof: Any = None, mst_pck: bool = True,
    irregulars: bool = True) -> None: ...

class Metadata:
    nlines: Any
    nsamples: Any
    exposure: Any
    filter: Any
    tstart: Any
    tstop: Any
    target: Any
    mode: Any
    window: Any
    window_origin: Any
    window_shape: Any
    window_uv_origin: Any
    window_uv_shape: Any
    def __init__(self, meta_dict: Any) -> None: ...
    def trim(self, data: Any, full_fov: bool = False) -> Any: ...
    def fov(self, full_fov: bool = False) -> Any: ...
class SSI:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    @staticmethod
    def initialize(planets: Any = None, asof: Any = None, mst_pck: bool = True,
        irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
