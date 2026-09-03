##########################################################################################
# oops/surface/spice_shape.pyi
##########################################################################################
"""Type stub for :mod:`oops.surface.spice_shape`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.frame.spiceframe import SpiceFrame as SpiceFrame
from oops.path.spicepath import SpicePath as SpicePath
from oops.surface.ellipsoid import Ellipsoid as Ellipsoid
from oops.surface.spheroid import Spheroid as Spheroid

def spice_shape(spice_id: Any, frame: Any = None, default_radii: Any = None) -> Any: ...

##########################################################################################
