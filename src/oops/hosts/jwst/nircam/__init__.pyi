##########################################################################################
# oops/hosts/jwst/nircam/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.jwst.nircam`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.jwst import JWST

__all__ = ['from_file', 'NIRCam']

def from_file(filespec: Any, **options: Any) -> Any: ...

class NIRCam(JWST):
    def header_subfields(self, hdulist: Any, **options: Any) -> Any: ...
    def filter_bandpass(self, hdulist: Any, **options: Any) -> Any: ...
    @staticmethod
    def from_hdulist(hdulist: Any, **options: Any) -> Any: ...

##########################################################################################
