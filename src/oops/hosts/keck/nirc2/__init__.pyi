##########################################################################################
# oops/hosts/keck/nirc2/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.keck.nirc2`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

import pathlib
from typing import Any
from astropy.io.fits import HDUList
from filecache import FCPath as FCPath
from oops.fov import FlatFOV
from oops.hosts.keck import Keck as Keck
from oops.observation import Snapshot

__all__ = ['from_file', 'NIRC2']

def from_file(filespec: str | pathlib.Path | FCPath, **parameters: Any) -> Snapshot: ...

class NIRC2(Keck):
    def filter_name(self, keck_file: HDUList) -> str: ...
    def define_fov(self, keck_file: HDUList, **parameters: Any) -> FlatFOV: ...
    @staticmethod
    def from_opened_fitsfile(keck_file: HDUList, **parameters: Any) -> Snapshot: ...

##########################################################################################
