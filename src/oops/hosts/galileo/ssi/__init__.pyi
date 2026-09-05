##########################################################################################
# oops/hosts/galileo/ssi/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.galileo.ssi`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from filecache import FCPath as FCPath
from oops import FOV as FOV, Path as Path
from numpy import ndarray

__all__ = ['from_file', 'from_index', 'initialize', 'Metadata', 'SSI']

def from_file(filespec: str | Path | FCPath, return_all_planets: bool = False,
    full_fov: bool = False, method: str = 'strict', **parameters: Any) -> Any: ...

def from_index(filespec: Any, supplemental_filespec: Any = None, full_fov: bool = False,
    **parameters: Any) -> Any: ...

def initialize(planets: list | None = None, asof: str | None = None,
    mst_pck: bool = True, irregulars: bool = True) -> None: ...

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
    def trim(self, data: Any, full_fov: bool = False) -> ndarray: ...
    def fov(self, full_fov: bool = False) -> FOV: ...

class SSI:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    @staticmethod
    def initialize(planets: list | None = None, asof: str | None = None,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
