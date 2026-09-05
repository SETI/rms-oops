##########################################################################################
# oops/hosts/hst/nicmos/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.hst.nicmos`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from oops.hosts.hst import HST as HST

__all__ = ['from_file', 'NICMOS']

def from_file(filespec: Any, **parameters: Any) -> Any: ...

class NICMOS(HST):
    def detector_name(self, hdulist: Any) -> Any: ...  # type: ignore[override]
    def filter_name(self, hdulist: Any) -> Any: ...
    def define_fov(self, hdulist: Any, **parameters: Any) -> Any: ...
    def select_syn_files(self, hdulist: Any, **parameters: Any) -> Any: ...
    def dn_per_sec_factor(self, hdulist: Any) -> Any: ...
    @staticmethod
    def from_hdulist(hdulist: Any, **parameters: Any) -> Any: ...

##########################################################################################
