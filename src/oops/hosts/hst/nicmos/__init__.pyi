##########################################################################################
# oops/hosts/hst/nicmos/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.hst.nicmos`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from pathlib import Path
from astropy.io.fits import HDUList
from filecache import FCPath as FCPath
from oops import FOV as FOV
from oops.hosts.hst import HST as HST
from oops.observation import Snapshot as Snapshot

__all__ = ['from_file', 'NICMOS']

def from_file(filespec: str | Path | FCPath, **parameters: Any) -> Snapshot: ...

class NICMOS(HST):
    def detector_name(self, hdulist: HDUList) -> str: ...  # type: ignore[override]
    def filter_name(self, hdulist: HDUList) -> str: ...
    def define_fov(self, hdulist: HDUList, **parameters: Any) -> FOV: ...
    def select_syn_files(self, hdulist: HDUList, **parameters: Any) -> list[str]: ...
    def dn_per_sec_factor(self, hdulist: HDUList) -> float: ...
    @staticmethod
    def from_hdulist(hdulist: HDUList, **parameters: Any) -> Snapshot: ...

##########################################################################################
