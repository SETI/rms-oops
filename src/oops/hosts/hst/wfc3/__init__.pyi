##########################################################################################
# oops/hosts/hst/wfc3/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.hst.wfc3`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from .. import HST

__all__ = ['from_file', 'WFC3']

def from_file(filespec: Any, **parameters: Any) -> Any: ...

class WFC3(HST):
    def filter_name(self, hdulist: Any) -> Any: ...
    @staticmethod
    def from_hdulist(hdulist: Any, **parameters: Any) -> Any: ...

##########################################################################################
