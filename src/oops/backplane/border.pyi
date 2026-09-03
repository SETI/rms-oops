##########################################################################################
# oops/backplane/border.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.border`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane

def border_above(self, backplane_key: Any, value: Any) -> Any: ...

def border_below(self, backplane_key: Any, value: Any) -> Any: ...

def border_atop(self, backplane_key: Any, value: Any) -> Any: ...

def border_inside(self, backplane_key: Any) -> Any: ...

def border_outside(self, backplane_key: Any) -> Any: ...

##########################################################################################
