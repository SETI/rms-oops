##########################################################################################
# oops/hosts/keck/nirc2.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.keck.nirc2`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from . import Keck as Keck

def from_file(filespec: Any, **parameters: Any) -> Any: ...

class NIRC2(Keck):
    def filter_name(self, keck_file: Any) -> Any: ...
    def define_fov(self, keck_file: Any, **parameters: Any) -> Any: ...
    @staticmethod
    def from_opened_fitsfile(keck_file: Any, **parameters: Any) -> Any: ...

##########################################################################################
