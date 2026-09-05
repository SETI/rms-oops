##########################################################################################
# oops/hosts/jwst/nircam/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.jwst.nircam`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from filecache import FCPath as FCPath
from oops import Path as Path
from oops.hosts.jwst import JWST as JWST

__all__ = ['from_file', 'NIRCam']

def from_file(filespec: str | Path | FCPath, **options: Any) -> Any: ...

class NIRCam(JWST):
    def header_subfields(self, hdulist: Any, **options: Any) -> Any: ...
    def filter_bandpass(self, hdulist: Any, **options: Any) -> Any: ...
    @staticmethod
    def from_hdulist(hdulist: Any, **options: Any) -> Any: ...

##########################################################################################
