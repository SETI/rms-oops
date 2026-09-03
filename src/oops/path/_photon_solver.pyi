##########################################################################################
# oops/path/_photon_solver.pyi
##########################################################################################
"""Type stub for :mod:`oops.path._photon_solver`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.config import LOGGING as LOGGING, PATH_PHOTONS as PATH_PHOTONS
from oops.constants import C as C
from oops.frame.frame_ import Frame as Frame
from oops.path.path_ import Path as Path

def photon_to_event(self, arrival: Any, *, derivs: bool = False, guess: Any = None,
    antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

def photon_from_event(self, departure: Any, *, derivs: bool = False, guess: Any = None,
    antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...

##########################################################################################
