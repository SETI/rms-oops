##########################################################################################
# oops/observation/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.observation`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.observation.insitu import InSitu as InSitu
from oops.observation.observation_ import Observation as Observation
from oops.observation.pixel import Pixel as Pixel
from oops.observation.rasterslit1d import RasterSlit1D as RasterSlit1D
from oops.observation.slit1d import Slit1D as Slit1D
from oops.observation.snapshot import Snapshot as Snapshot
from oops.observation.timedimage import TimedImage as TimedImage

__all__ = ['Observation', 'InSitu', 'Pixel', 'RasterSlit1D', 'Slit1D', 'Snapshot',
           'TimedImage']

##########################################################################################
