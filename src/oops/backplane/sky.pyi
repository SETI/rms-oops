##########################################################################################
# oops/backplane/sky.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.sky`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.frame import Frame as Frame

def right_ascension(self, event_key: Any = (), apparent: bool = True,
    direction: str = 'arr') -> Any: ...

def declination(self, event_key: Any = (), apparent: bool = True,
    direction: str = 'arr') -> Any: ...

def celestial_north_angle(self, event_key: Any = ()) -> Any: ...

def celestial_east_angle(self, event_key: Any = ()) -> Any: ...

def center_right_ascension(self, event_key: Any, apparent: bool = True,
    direction: str = 'arr') -> Any: ...

def center_declination(self, event_key: Any, apparent: bool = True,
    direction: str = 'arr') -> Any: ...

##########################################################################################
