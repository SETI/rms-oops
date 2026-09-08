##########################################################################################
# oops/hosts/galileo/ssi/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.galileo.ssi`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

from pathlib import Path
from typing import Any
from filecache import FCPath as FCPath
from oops import FOV as FOV
from oops.observation import Snapshot as Snapshot
from numpy import ndarray

__all__ = ['from_file', 'from_index', 'initialize', 'Metadata', 'SSI']

def from_file(filespec: str | Path | FCPath, return_all_planets: bool = False,
    full_fov: bool = False, method: str = 'strict', **parameters: Any) -> Snapshot: ...

def from_index(filespec: str | Path | FCPath,
    supplemental_filespec: str | Path | FCPath | None = None, full_fov: bool = False,
    **parameters: Any) -> list[Snapshot]: ...

def initialize(planets: list | None = None, asof: str | None = None,
    mst_pck: bool = True, irregulars: bool = True) -> None: ...

class Metadata:
    nlines: int
    nsamples: int
    exposure: float
    filter: str
    tstart: float
    tstop: float
    target: str
    mode: str
    window: ndarray | None
    window_origin: ndarray
    window_shape: ndarray
    window_uv_origin: ndarray
    window_uv_shape: ndarray
    def __init__(self, meta_dict: dict) -> None: ...
    def trim(self, data: ndarray, full_fov: bool = False) -> ndarray: ...
    def fov(self, full_fov: bool = False) -> FOV: ...

class SSI:
    instrument_kernel: dict | None
    fovs: dict[str, FOV]
    initialized: bool
    @staticmethod
    def initialize(planets: list | None = None, asof: str | None = None,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
