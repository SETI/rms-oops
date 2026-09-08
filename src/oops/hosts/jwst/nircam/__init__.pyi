##########################################################################################
# oops/hosts/jwst/nircam/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.jwst.nircam`.

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
from oops.hosts.jwst import JWST as JWST
from oops.observation import TimedImage
from tabulation import Tabulation

__all__ = ['from_file', 'NIRCam']

def from_file(filespec: str | pathlib.Path | FCPath, **options: Any) -> TimedImage: ...

class NIRCam(JWST):
    def header_subfields(self, hdulist: HDUList, **options: Any) -> dict[str, Any]: ...
    def filter_bandpass(self, hdulist: HDUList, **options: Any) -> Tabulation: ...
    @staticmethod
    def from_hdulist(hdulist: HDUList, **options: Any) -> TimedImage: ...

##########################################################################################
