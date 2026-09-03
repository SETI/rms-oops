##########################################################################################
# oops/body.pyi
##########################################################################################
"""Type stub for :mod:`oops.body`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.frame.frame_ import Frame as Frame
from oops.frame.poleframe import PoleFrame as PoleFrame
from oops.frame.ringframe import RingFrame as RingFrame
from oops.frame.spiceframe import SpiceFrame as SpiceFrame
from oops.frame.synchronousframe import SynchronousFrame as SynchronousFrame
from oops.frame.twovectorframe import TwoVectorFrame as TwoVectorFrame
from oops.gravity.gravity_ import Gravity as Gravity
from oops.gravity.oblategravity import OblateGravity as OblateGravity
from oops.oops import Oops as Oops
from oops.path.multipath import MultiPath as MultiPath
from oops.path.path_ import Path as Path
from oops.path.spicepath import SpicePath as SpicePath
from oops.surface.nullsurface import NullSurface as NullSurface
from oops.surface.orbitplane import OrbitPlane as OrbitPlane
from oops.surface.ringplane import RingPlane as RingPlane
from oops.surface.spice_shape import spice_shape as spice_shape

JUPITER_ALIASES: Any
SATURN_ALIASES: Any
ALIASES: Any

def lrange(*args: Any) -> Any: ...
MARS_ALL_MOONS: Any
JUPITER_CLASSICAL: Any
JUPITER_REGULAR: Any
JUPITER_IRREGULAR: Any
JUPITER_MAIN_RING_LIMIT: float
SATURN_CLASSICAL_INNER: Any
SATURN_CLASSICAL_OUTER: Any
SATURN_CLASSICAL_IRREG: Any
SATURN_REGULAR: Any
SATURN_IRREGULAR: Any
SATURN_MAIN_RINGS: Any
SATURN_D_RING: Any
SATURN_C_RING: Any
SATURN_B_RING: Any
SATURN_A_RING: Any
SATURN_F_RING_CORE: float
SATURN_F_RING_LIMIT: float
SATURN_RINGS: Any
SATURN_AB_RINGS: Any
URANUS_CLASSICAL: Any
URANUS_INNER: Any
URANUS_IRREGULAR: Any
URANUS_EPSILON_LIMIT: float
URANUS_MU_LIMIT: Any
URANUS_NU_LIMIT: Any
URANUS_OLD_GRAVITY: Any
URANUS_SIX_ELEMENTS: Any
URANUS_FIVE_ELEMENTS: Any
URANUS_FOUR_ELEMENTS: Any
URANUS_ALPHA_ELEMENTS: Any
URANUS_BETA_ELEMENTS: Any
URANUS_ETA_ELEMENTS: Any
URANUS_GAMMA_ELEMENTS: Any
URANUS_DELTA_ELEMENTS: Any
URANUS_LAMBDA_ELEMENTS: Any
URANUS_EPSILON_ELEMENTS: Any
NEPTUNE_CLASSICAL_INNER: Any
NEPTUNE_CLASSICAL_OUTER: Any
NEPTUNE_REGULAR: Any
NEPTUNE_IRREGULAR: Any
NEPTUNE_ADAMS_LIMIT: float
NEPTUNE_INVARIABLE_RA: Any
NEPTUNE_INVARIABLE_DEC: Any
CHARON: Any
PLUTO_REGULAR: Any
PLUTO_RADIUS: float
CHARON_RADIUS: float
PLUTO_CHARON_DISTANCE: float

class Body(Oops):
    BODY_REGISTRY: Any
    STANDARD_BODIES: Any
    name: Any
    is_standard: bool
    spk: Any
    spice_name: Any
    spice_id: Any
    path: Any
    frame: Any
    ring_frame: Any
    ring_epoch: Any
    ring_is_retrograde: bool
    ring_pole: Any
    ring_body: Any
    is_ring: bool
    invariable_pole: Any
    invariable_frame: Any
    parent: Any
    barycenter: Any
    surface: Any
    radius: float
    inner_radius: float
    gravity: Any
    lightsource: Any
    keywords: Any
    child_names: Any
    is_registered: bool
    def __init__(self, name: Any, path: Any, frame: Any, parent: Any = None,
        barycenter: Any = None, spice_name: Any = None) -> None: ...
    def apply_surface(self, surface: Any, radius: Any,
        inner_radius: float = 0.0) -> None: ...
    def apply_ring_frame(self, epoch: Any = None, retrograde: bool = False,
        pole: Any = None) -> None: ...
    def apply_gravity(self, gravity: Any) -> None: ...
    def add_keywords(self, keywords: Any) -> None: ...
    @property
    def children(self) -> Any: ...
    def select_children(self, include_all: Any = None, include_any: Any = None,
        exclude: Any = None, radius: Any = None, recursive: bool = False) -> Any: ...
    @staticmethod
    def name_in(bodies: Any, names: Any) -> Any: ...
    @staticmethod
    def name_not_in(bodies: Any, names: Any) -> Any: ...
    @staticmethod
    def radius_in_range(bodies: Any, min: Any, max: Any = ...) -> Any: ...
    @staticmethod
    def radius_not_in_range(bodies: Any, min: Any, max: Any = ...) -> Any: ...
    @staticmethod
    def surface_class_in(bodies: Any, class_names: Any) -> Any: ...
    @staticmethod
    def surface_class_not_in(bodies: Any, class_names: Any) -> Any: ...
    @staticmethod
    def has_gravity(bodies: Any) -> Any: ...
    @staticmethod
    def has_no_gravity(bodies: Any) -> Any: ...
    @staticmethod
    def has_children(bodies: Any) -> Any: ...
    @staticmethod
    def has_no_children(bodies: Any) -> Any: ...
    @staticmethod
    def has_ring(bodies: Any) -> Any: ...
    @staticmethod
    def has_no_ring(bodies: Any) -> Any: ...
    @staticmethod
    def keywords_include_any(bodies: Any, keywords: Any) -> Any: ...
    @staticmethod
    def keywords_include_all(bodies: Any, keywords: Any) -> Any: ...
    @staticmethod
    def keywords_do_not_include(bodies: Any, keywords: Any) -> Any: ...
    @staticmethod
    def define_multipath(bodies: Any, origin: str = 'SSB', frame: str = 'J2000',
        path_id: Any = None) -> Any: ...
    @staticmethod
    def lookup(key: Any) -> Any: ...
    @staticmethod
    def exists(key: Any) -> Any: ...
    @staticmethod
    def as_body(body: Any) -> Any: ...
    @staticmethod
    def as_body_name(body: Any) -> Any: ...
    @staticmethod
    def reset_registry() -> None: ...
    def as_path(self) -> Any: ...
    def photon_to_event(self, event: Any, derivs: bool = False, guess: Any = None,
        antimask: Any = None, quick: Any = None, converge: Any = None) -> Any: ...
    @staticmethod
    def define_solar_system(start_time: Any = None, stop_time: Any = None,
        asof: Any = None, **args: Any) -> Any: ...
    MARS_MOONS_LOADED: Any
    JUPITER_MOONS_LOADED: Any
    SATURN_MOONS_LOADED: Any
    URANUS_MOONS_LOADED: Any
    NEPTUNE_MOONS_LOADED: Any
    PLUTO_MOONS_LOADED: Any
    @staticmethod
    def define_bodies(spice_ids: Any, parent: Any, barycenter: Any, keywords: Any,
        is_standard: bool = False) -> None: ...
    @staticmethod
    def define_body(spice_id: Any, parent: Any, barycenter: Any, keywords: Any,
        is_standard: bool = False, name: Any = None) -> None: ...
    @staticmethod
    def define_ring(parent_name: Any, ring_name: Any, radii: Any, keywords: Any,
        retrograde: bool = False, barycenter_name: Any = None, pole: Any = None,
        is_standard: bool = False) -> Any: ...
    @staticmethod
    def define_orbit(parent_name: Any, ring_name: Any, elements: Any, epoch: Any,
        reference: Any, keywords: Any, is_standard: bool = False) -> None: ...
    @staticmethod
    def define_small_body(spice_id: Any, name: Any = None, spk: Any = None,
        keywords: Any = None, parent: str = 'SUN', barycenter: str = 'SSB',
        is_standard: bool = False) -> None: ...

##########################################################################################
