##########################################################################################
# oops/backplane/resolution.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.resolution`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.surface import Surface as Surface

def resolution(self, event_key: Any, axis: str = 'u') -> Any: ...

def center_resolution(self, event_key: Any, axis: str = 'u') -> Any: ...

def finest_resolution(self, event_key: Any) -> Any: ...

def coarsest_resolution(self, event_key: Any) -> Any: ...

##########################################################################################
