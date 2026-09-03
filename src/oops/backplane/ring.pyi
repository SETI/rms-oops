##########################################################################################
# oops/backplane/ring.pyi
##########################################################################################
"""Type stub for :mod:`oops.backplane.ring`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.backplane import Backplane as Backplane
from oops.body import Body as Body
from oops.frame import Frame as Frame

RING_BACKPLANES: Any

def ring_radius(self, event_key: Any, rmin: Any = None, rmax: Any = None) -> Any: ...

def ring_longitude(self, event_key: Any, reference: str = 'node') -> Any: ...

def radial_mode(self, backplane_key: Any, cycles: Any, epoch: Any, amp: Any, peri0: Any,
    speed: Any, a0: float = 0.0, dperi_da: float = 0.0,
    reference: str = 'node') -> Any: ...

def ring_azimuth(self, event_key: Any, direction: str = 'obs',
    apparent: bool = True) -> Any: ...

def ring_elevation(self, event_key: Any, direction: str = 'obs', pole: str = 'prograde',
    apparent: bool = True) -> Any: ...

def ring_incidence_angle(self, event_key: Any, pole: str = 'sunward',
    apparent: bool = True) -> Any: ...

def ring_emission_angle(self, event_key: Any, pole: str = 'sunward',
    apparent: bool = True) -> Any: ...

def ring_sub_observer_longitude(self, event_key: Any, reference: str = 'node') -> Any: ...

def ring_sub_solar_longitude(self, event_key: Any, reference: str = 'node') -> Any: ...

def ring_center_incidence_angle(self, event_key: Any, pole: str = 'sunward',
    apparent: bool = True) -> Any: ...

def ring_center_emission_angle(self, event_key: Any, pole: str = 'sunward',
    apparent: bool = True) -> Any: ...

def ring_radial_resolution(self, event_key: Any) -> Any: ...

def ring_angular_resolution(self, event_key: Any, units: str = 'rad') -> Any: ...

def ring_gradient_angle(self, event_key: Any) -> Any: ...

def ring_shadow_radius(self, event_key: Any, ring_surface_key: Any) -> Any: ...

def ring_shadow_incidence(self, event_key: Any, ring_surface_key: Any) -> Any: ...

def ring_radius_in_front(self, event_key: Any, ring_surface_key: Any) -> Any: ...

##########################################################################################
