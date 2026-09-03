##########################################################################################
# oops/fov/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.fov`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.fov.barrelfov import BarrelFOV as BarrelFOV
from oops.fov.flatfov import FlatFOV as FlatFOV
from oops.fov.fov_ import FOV as FOV
from oops.fov.gapfov import GapFOV as GapFOV
from oops.fov.nullfov import NullFOV as NullFOV
from oops.fov.offsetfov import OffsetFOV as OffsetFOV
from oops.fov.platescale import Platescale as Platescale
from oops.fov.polynomialfov import PolynomialFOV as PolynomialFOV
from oops.fov.slicefov import SliceFOV as SliceFOV
from oops.fov.subarray import Subarray as Subarray
from oops.fov.subsampledfov import SubsampledFOV as SubsampledFOV
from oops.fov.tdifov import TDIFOV as TDIFOV
from oops.fov.wcsfov import WCSFOV as WCSFOV

__all__ = ['FOV', 'BarrelFOV', 'FlatFOV', 'GapFOV', 'NullFOV', 'OffsetFOV', 'Platescale',
           'PolynomialFOV', 'SliceFOV', 'Subarray', 'SubsampledFOV', 'TDIFOV', 'WCSFOV']

##########################################################################################
