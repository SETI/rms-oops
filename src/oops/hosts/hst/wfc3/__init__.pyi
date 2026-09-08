##########################################################################################
# oops/hosts/hst/wfc3/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.hst.wfc3`.

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
from oops.hosts.hst import HST as HST
from oops.observation import Snapshot as Snapshot

__all__ = ['from_file', 'WFC3']

def from_file(filespec: str | Path | FCPath, **parameters: Any) -> Snapshot: ...

class WFC3(HST):
    def filter_name(self, hdulist: HDUList) -> str: ...
    @staticmethod
    def from_hdulist(hdulist: HDUList, **parameters: Any) -> Snapshot: ...

##########################################################################################
