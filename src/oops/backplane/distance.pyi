##########################################################################################
# oops/backplane/distance.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.distance`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.constants import C as C

def distance(self, event_key: Any, direction: str = 'dep') -> Any: ...

def light_time(self, event_key: Any, direction: str = 'dep') -> Any: ...

def event_time(self, event_key: Any) -> Any: ...

def center_distance(self, event_key: Any, direction: str = 'dep') -> Any: ...

def center_light_time(self, event_key: Any, direction: str = 'dep') -> Any: ...

def center_time(self, event_key: Any) -> Any: ...

##########################################################################################
