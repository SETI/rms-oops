##########################################################################################
# oops/hosts/hst/nicmos/nic1.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.hst.nicmos.nic1`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from . import NICMOS as NICMOS

def from_file(filespec: Any, **parameters: Any) -> Any: ...

class NIC1(NICMOS):
    DETECTOR_SYN_FILES: Any
    FILTER_SYN_FILE_PARTS: Any
    @staticmethod
    def from_hdulist(hdulist: Any, **parameters: Any) -> Any: ...

##########################################################################################
