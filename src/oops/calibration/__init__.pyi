##########################################################################################
# oops/calibration/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.calibration`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.calibration.calibration_ import Calibration as Calibration
from oops.calibration.flatcalib import FlatCalib as FlatCalib
from oops.calibration.nullcalib import NullCalib as NullCalib
from oops.calibration.radiance import Radiance as Radiance
from oops.calibration.rawcounts import RawCounts as RawCounts

__all__ = ['Calibration', 'FlatCalib', 'NullCalib', 'Radiance', 'RawCounts']

##########################################################################################
