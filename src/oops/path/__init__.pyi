##########################################################################################
# oops/path/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.path`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.path.circlepath import CirclePath as CirclePath
from oops.path.coordpath import CoordPath as CoordPath
from oops.path.fixedpath import FixedPath as FixedPath
from oops.path.keplerpath import KeplerPath as KeplerPath
from oops.path.linearcoordpath import LinearCoordPath as LinearCoordPath
from oops.path.linearpath import LinearPath as LinearPath
from oops.path.multipath import MultiPath as MultiPath
from oops.path.path_ import (LinkedPath as LinkedPath, NullPath as NullPath, Path as Path,
                             RelativePath as RelativePath, ReversedPath as ReversedPath,
                             RotatedPath as RotatedPath, SSBPath as SSBPath)
from oops.path.pathshift import PathShift as PathShift
from oops.path.quickpath import QuickPath as QuickPath
from oops.path.spicepath import SpicePath as SpicePath

__all__ = ['Path', 'NullPath', 'SSBPath', 'LinkedPath', 'RelativePath', 'ReversedPath',
           'RotatedPath', 'CirclePath', 'CoordPath', 'FixedPath', 'KeplerPath',
           'LinearCoordPath', 'LinearPath', 'MultiPath', 'PathShift', 'QuickPath',
           'SpicePath']

##########################################################################################
