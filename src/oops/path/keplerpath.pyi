##########################################################################################
# oops/path/keplerpath.pyi
##########################################################################################
"""Type stub for :mod:`oops.path.keplerpath`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.cache import Cache as Cache
from oops.event import Event as Event
from oops.fittable import Fittable as Fittable
from oops.frame.frame_ import Frame as Frame
from oops.path.path_ import Path as Path

class KeplerPath(Path, Fittable):
    nparams: Any
    def __init__(self, body: Any, epoch: Any, elements: Any = None, observer: Any = None,
        *, wobbles: Any = (), path_id: Any = None) -> None: ...
    def set_elements(self, elements: Any) -> None: ...
    def get_elements(self) -> Any: ...
    @property
    def params(self) -> Any: ...
    def copy(self) -> Any: ...
    def event_at_time(self, time: Any, *, quick: Any = None,
        partials: bool = False) -> Any: ...
    def node_at_time(self, time: Any) -> Any: ...
    def pole_at_time(self, time: Any) -> Any: ...
    def photon_to_event(self, arrival: Any, *, derivs: bool = False, guess: Any = None,
        antimask: Any = None, quick: Any = None, converge: Any = None,
        partials: bool = False) -> Any: ...

##########################################################################################
