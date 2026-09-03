##########################################################################################
# oops/surface/_photon_solver.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface._photon_solver`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING, SURFACE_PHOTONS as SURFACE_PHOTONS
from oops.constants import C as C
from oops.event import Event as Event
from oops.frame.frame_ import Frame as Frame
from oops.path.path_ import Path as Path

DEBUG: bool

def photon_to_event(self, arrival: Any, *, derivs: bool = False, guess: Any = None,
    antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

def photon_from_event(self, departure: Any, *, derivs: bool = False, guess: Any = None,
    antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

def photon_to_coords(self, arrival: Any, coords: Any, *, derivs: bool = False,
    guess: Any = None, antimask: Any = None, quick: Any = None,
    converge: Any = None) -> Any: ...

def photon_from_coords(self, departure: Any, coords: Any, *, derivs: bool = False,
    guess: Any = None, antimask: Any = None, quick: Any = None,
    converge: Any = None) -> Any: ...

def photon_normal_to_event(self, arrival: Any, *, derivs: bool = False, guess: Any = None,
    antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

def photon_event_to_normal(self, departure: Any, *, derivs: bool = False,
    guess: Any = None, antimask: Any = None, quick: Any = None,
    converge: Any = None) -> Any: ...

def photon_path_to_normal(self, time: Any, path: Any, *, derivs: bool = False,
    guess: Any = None, antimask: Any = None, quick: Any = None,
    converge: Any = None) -> Any: ...

def photon_normal_to_path(self, time: Any, path: Any, *, derivs: bool = False,
    guess: Any = None, antimask: Any = None, quick: Any = None,
    converge: Any = None) -> Any: ...

##########################################################################################
