##########################################################################################
# oops/backplane/ansa.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.ansa`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane

ANSA_BACKPLANES: Any

def ansa_radius(self, event_key: Any, radius_type: str = 'positive',
    rmax: Any = None) -> Any: ...

def ansa_altitude(self, event_key: Any) -> Any: ...

def ansa_longitude(self, event_key: Any, reference: str = 'node') -> Any: ...

def ansa_radial_resolution(self, event_key: Any) -> Any: ...

def ansa_vertical_resolution(self, event_key: Any) -> Any: ...

##########################################################################################
