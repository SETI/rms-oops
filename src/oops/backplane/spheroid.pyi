##########################################################################################
# oops/backplane/spheroid.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.spheroid`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane

def longitude(self, event_key: Any, reference: str = 'iau', direction: str = 'west',
    minimum: int = 0, lon_type: str = 'centric') -> Any: ...

def latitude(self, event_key: Any, lat_type: str = 'centric') -> Any: ...

def sub_observer_longitude(self, event_key: Any, reference: str = 'iau',
    direction: str = 'west', minimum: int = 0) -> Any: ...

def sub_solar_longitude(self, event_key: Any, reference: str = 'iau',
    direction: str = 'west', minimum: int = 0) -> Any: ...

def sub_observer_latitude(self, event_key: Any, lat_type: str = 'centric') -> Any: ...

def sub_solar_latitude(self, event_key: Any, lat_type: str = 'centric') -> Any: ...

##########################################################################################
