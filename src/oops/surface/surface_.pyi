##########################################################################################
# oops/surface/surface_.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.surface_`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.event import Event as Event
from oops.mutable import Mutable as Mutable

class Surface(Mutable):
    IS_VIRTUAL: bool
    IS_TIME_DEPENDENT: bool
    HAS_INTERIOR: bool
    COORDINATE_TYPE: str
    COORDINATE_NAMES: Any
    COORDINATE_ABBREVS: Any
    COORDINATE_RANGES: Any
    def coords_from_vector3(self, pos: Any, *, obs: Any = None, time: Any = None,
        axes: int = 2, derivs: bool = False, hints: Any = None) -> Any: ...
    def vector3_from_coords(self, coords: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None) -> Any: ...
    def intercept(self, obs: Any, los: Any, *, time: Any = None, direction: str = 'dep',
        derivs: bool = False, guess: Any = None, hints: Any = None) -> Any: ...
    def normal(self, pos: Any, *, obs: Any = None, time: Any = None, derivs: bool = False,
        hints: Any = None) -> Any: ...
    def intercept_with_normal(self, normal: Any, *, obs: Any = None, time: Any = None,
        derivs: bool = False, hints: Any = None) -> Any: ...
    def intercept_normal_to(self, pos: Any, *, obs: Any = None, time: Any = None,
        direction: str = 'dep', derivs: bool = False, guess: Any = None,
        hints: Any = None) -> Any: ...
    def velocity(self, pos: Any, *, obs: Any = None, time: Any = None) -> Any: ...
    def position_is_inside(self, pos: Any, *, obs: Any = None,
        time: Any = None) -> Any: ...
    def reference(self) -> Any: ...
    def coords_of_event(self, event: Any, *, obs: Any = None, axes: int = 3,
        derivs: bool = False) -> Any: ...
    def apply_coords_to_event(self, event: Any, *, obs: Any = None, axes: int = 3,
        derivs: bool = True) -> Any: ...
    def event_at_coords(self, time: Any, coords: Any, *, obs: Any = None,
        derivs: bool = False) -> Any: ...
    @staticmethod
    def resolution(dpos_duv: Any) -> Any: ...
    def photon_event_to_normal(self, departure: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_from_coords(self, departure: Any, coords: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_from_event(self, departure: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_normal_to_event(self, arrival: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_normal_to_path(self, time: Any, path: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_path_to_normal(self, time: Any, path: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_to_coords(self, arrival: Any, coords: Any, *, derivs: bool = False,
        guess: Any = None, antimask: Any = None, quick: Any = None,
        converge: Any = None) -> Any: ...
    def photon_to_event(self, arrival: Any, *, derivs: bool = False, guess: Any = None,
        antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

##########################################################################################
