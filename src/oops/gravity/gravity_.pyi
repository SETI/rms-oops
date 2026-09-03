##########################################################################################
# oops/gravity/gravity_.pyi
##########################################################################################
"""Type stub for :mod:`oops.gravity.gravity_`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.gravity.oblategravity import OblateGravity
from oops.oops import Oops as Oops

class Gravity(Oops):
    GRAVITY_REGISTRY: dict[str, OblateGravity]
    def potential(self, a: Any) -> Any: ...
    def omega(self, a: Any, *, e: float = 0.0, sin_i: float = 0.0) -> Any: ...
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
    @staticmethod
    def lookup(key: Any) -> Any: ...
    @staticmethod
    def exists(key: Any) -> Any: ...
    ARIEL: OblateGravity
    CALLISTO: OblateGravity
    CHARON: OblateGravity
    DIONE: OblateGravity
    EARTH: OblateGravity
    ENCELADUS: OblateGravity
    EUROPA: OblateGravity
    GANYMEDE: OblateGravity
    HYPERION: OblateGravity
    IAPETUS: OblateGravity
    IO: OblateGravity
    JUPITER: OblateGravity
    JUPITER_BARYCENTER: OblateGravity
    JUPITER_GALILEANS: OblateGravity
    MARS: OblateGravity
    MERCURY: OblateGravity
    MIMAS: OblateGravity
    MIRANDA: OblateGravity
    MOON: OblateGravity
    NEPTUNE: OblateGravity
    NEPTUNE_BARYCENTER: OblateGravity
    NEREID: OblateGravity
    OBERON: OblateGravity
    PHOEBE: OblateGravity
    PLUTO: OblateGravity
    PLUTO_BARYCENTER: OblateGravity
    PLUTO_CHARON: OblateGravity
    PLUTO_ONLY: OblateGravity
    RHEA: OblateGravity
    SATURN: OblateGravity
    SATURN_BARYCENTER: OblateGravity
    SATURN_TITAN: OblateGravity
    SSB: OblateGravity
    SUN: OblateGravity
    SUN_JUPITER: OblateGravity
    TETHYS: OblateGravity
    TITAN: OblateGravity
    TITANIA: OblateGravity
    TRITON: OblateGravity
    UMBRIEL: OblateGravity
    URANUS: OblateGravity
    URANUS_BARYCENTER: OblateGravity
    VENUS: OblateGravity

##########################################################################################
