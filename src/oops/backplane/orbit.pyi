##########################################################################################
# oops/backplane/orbit.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.orbit`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.body import Body as Body
from oops.frame import Frame as Frame

def orbit_longitude(self, event_key: Any, reference: str = 'obs',
    planet: Any = None) -> Any: ...

##########################################################################################
