##########################################################################################
# oops/hosts/hst/acs/sbc.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.hst.acs.sbc`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from . import ACS as ACS

def from_file(filespec: Any, **parameters: Any) -> Any: ...
IDC_DICT: Any
GENERAL_SYN_FILES: Any
FILTER_SYN_FILE: Any

class SBC(ACS):
    def define_fov(self, hdulist: Any, **parameters: Any) -> Any: ...
    def filter_name(self, hdulist: Any, layer: Any = None) -> Any: ...
    def select_syn_files(self, hdulist: Any, **parameters: Any) -> Any: ...
    @staticmethod
    def from_hdulist(hdulist: Any, **parameters: Any) -> Any: ...

##########################################################################################
