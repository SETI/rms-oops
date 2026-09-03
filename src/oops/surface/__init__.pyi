##########################################################################################
# oops/surface/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.surface.ansa import Ansa as Ansa
from oops.surface.centricellipsoid import CentricEllipsoid as CentricEllipsoid
from oops.surface.centricspheroid import CentricSpheroid as CentricSpheroid
from oops.surface.ellipsoid import Ellipsoid as Ellipsoid
from oops.surface.graphicellipsoid import GraphicEllipsoid as GraphicEllipsoid
from oops.surface.graphicspheroid import GraphicSpheroid as GraphicSpheroid
from oops.surface.limb import Limb as Limb
from oops.surface.nullsurface import NullSurface as NullSurface
from oops.surface.orbitplane import OrbitPlane as OrbitPlane
from oops.surface.polarlimb import PolarLimb as PolarLimb
from oops.surface.ringplane import RingPlane as RingPlane
from oops.surface.spheroid import Spheroid as Spheroid
from oops.surface.spice_shape import spice_shape as spice_shape
from oops.surface.surface_ import Surface as Surface

__all__ = ['Surface', 'Ansa', 'CentricEllipsoid', 'CentricSpheroid', 'Ellipsoid',
           'GraphicEllipsoid', 'GraphicSpheroid', 'Limb', 'NullSurface', 'OrbitPlane',
           'PolarLimb', 'RingPlane', 'Spheroid', 'spice_shape']

##########################################################################################
