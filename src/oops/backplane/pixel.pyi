##########################################################################################
# oops/backplane/pixel.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.pixel`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.body import Body as Body
from oops.constants import C as C

def body_diameter_in_pixels(self, event_key: Any, radius: int = 0,
    axis: str = 'max') -> Any: ...

def center_coordinate(self, event_key: Any, axis: str = 'u') -> Any: ...

##########################################################################################
