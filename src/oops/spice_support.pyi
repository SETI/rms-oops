##########################################################################################
# oops/spice_support.pyi
##########################################################################################
"""Type stub for :mod:`oops.spice_support`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.path.path_ import Path as Path

FRAME_TRANSLATION: Any
PATH_TRANSLATION: Any
LSK_LOADED: bool

def load_leap_seconds() -> None: ...

def body_id_and_name(arg: Any) -> Any: ...

def frame_id_and_name(arg: Any) -> Any: ...

def initialize() -> None: ...

##########################################################################################
