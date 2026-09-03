##########################################################################################
# oops/cadence/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.cadence`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.cadence.cadence_ import Cadence as Cadence
from oops.cadence.dualcadence import DualCadence as DualCadence
from oops.cadence.instant import Instant as Instant
from oops.cadence.metronome import Metronome as Metronome
from oops.cadence.reshapedcadence import ReshapedCadence as ReshapedCadence
from oops.cadence.reversedcadence import ReversedCadence as ReversedCadence
from oops.cadence.sequence import Sequence as Sequence
from oops.cadence.snapcadence import SnapCadence as SnapCadence
from oops.cadence.tdicadence import TDICadence as TDICadence
from oops.cadence.timeshift import TimeShift as TimeShift

__all__ = ['Cadence', 'DualCadence', 'Instant', 'Metronome', 'ReshapedCadence',
           'ReversedCadence', 'Sequence', 'SnapCadence', 'TDICadence', 'TimeShift']

##########################################################################################
