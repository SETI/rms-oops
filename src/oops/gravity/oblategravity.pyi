##########################################################################################
# oops/gravity/oblategravity.pyi
##########################################################################################
"""Type stub for :mod:`oops.gravity.oblategravity`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.gravity import Gravity as Gravity

class OblateGravity(Gravity):
    gm: Any
    jn: Any
    rp: Any
    r2: Any
    potential_jn: Any
    omega_jn: Any
    kappa_jn: Any
    nu_jn: Any
    domega_jn: Any
    dkappa_jn: Any
    dnu_jn: Any
    def __init__(self, gm: Any, jlist: Any = (), radius: float = 1.0) -> None: ...
    def potential(self, a: Any) -> Any: ...
    def omega(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def kappa2(self, a: Any) -> Any: ...
    def kappa(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def nu(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def domega_da(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def dkappa_da(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def dnu_da(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def combo(self, a: Any, factors: Any, *, e: float = 0.0,
        sin_i: float = 0.0) -> Any: ...
    def dcombo_da(self, a: Any, factors: Any, *, e: float = 0.0,
        sin_i: float = 0.0) -> Any: ...
    def solve_a(self, freq: Any, factors: Any = (1, 0, 0), *, e: float = 0.0,
        sin_i: float = 0.0) -> Any: ...
    def n(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def dmean_dt(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def dperi_dt(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def dnode_dt(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def d_dmean_dt_da(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def d_dperi_dt_da(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def d_dnode_dt_da(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
    def ilr_pattern(self, n: Any, m: Any, *, p: int = 1) -> Any: ...
    def olr_pattern(self, n: Any, m: Any, *, p: int = 1) -> Any: ...
    def state_from_osc(self, elements: Any, body_gm: float = 0.0) -> Any: ...
    def osc_from_state(self, pos: Any, vel: Any, body_gm: float = 0.0) -> Any: ...
    def state_from_geom(self, elements: Any, body_gm: float = 0.0) -> Any: ...
    def geom_from_state(self, pos: Any, vel: Any, body_gm: float = 0.0,
        tol: float = 1e-06) -> Any: ...
G_MKS: float
G_CGS: float
G_PER_KG: Any
G_PER_G: Any
SUN: Any
MERCURY: Any
VENUS: Any
EARTH: Any
MARS: Any
JUPITER_V1: Any
SATURN_V1: Any
URANUS_V1: Any
NEPTUNE_V1: Any
JUPITER: Any
SATURN: Any
URANUS: Any
NEPTUNE: Any
PLUTO_ONLY: Any
PLUTO = PLUTO_ONLY
MOON: Any
IO: Any
EUROPA: Any
GANYMEDE: Any
CALLISTO: Any
MIMAS: Any
ENCELADUS: Any
TETHYS: Any
DIONE: Any
RHEA: Any
TITAN: Any
HYPERION: Any
IAPETUS: Any
PHOEBE: Any
MIRANDA: Any
ARIEL: Any
UMBRIEL: Any
TITANIA: Any
OBERON: Any
TRITON: Any
NEREID: Any
CHARON: Any
SUN_JUPITER: Any
JUPITER_GALS: Any
SATURN_TITAN: Any
PLUTO_CHARON_OLD: Any
PLUTO_A: Any
CHARON_A: Any
ratio2: Any
gm1: Any
gm2: Any
PLUTO_CHARON_AS_RINGS: Any
PLUTO_CHARON = PLUTO_CHARON_AS_RINGS
name: Any

##########################################################################################
