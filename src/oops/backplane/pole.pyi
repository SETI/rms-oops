##########################################################################################
# oops/backplane/pole.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.pole`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.frame import Frame as Frame

def pole_clock_angle(self, event_key: Any) -> Any: ...

def pole_position_angle(self, event_key: Any) -> Any: ...

##########################################################################################
