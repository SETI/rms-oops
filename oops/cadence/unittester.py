###############################################################################
# oops/cadence/unittester.py
################################################################################

import unittest

from oops.cadence                 import Test_Cadence
from oops.cadence.dualcadence     import Test_DualCadence
from oops.cadence.instant         import Test_Instant
from oops.cadence.metronome       import Test_Metronome
from oops.cadence.reshapedcadence import Test_ReshapedCadence
from oops.cadence.reversedcadence import Test_ReversedCadence
from oops.cadence.sequence        import Test_Sequence
from oops.cadence.tdicadence      import Test_TDICadence

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
