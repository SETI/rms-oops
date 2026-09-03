##########################################################################################
# oops/backplane/limb.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.limb`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.surface.polarlimb import PolarLimb as PolarLimb

LIMB_BACKPLANES: Any

def limb_altitude(self, event_key: Any, zmin: Any = None, zmax: Any = None,
    scaled: bool = False) -> Any: ...

def limb_longitude(self, event_key: Any, reference: str = 'iau', direction: str = 'west',
    minimum: int = 0, lon_type: str = 'centric') -> Any: ...

def limb_latitude(self, event_key: Any, lat_type: str = 'centric') -> Any: ...

def limb_clock_angle(self, event_key: Any) -> Any: ...

##########################################################################################
