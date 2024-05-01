###############################################################################
# tests/cadence/unittester.py
################################################################################

import unittest

from tests.cadence.test_dualcadence     import Test_DualCadence
from tests.cadence.test_metronome       import Test_Metronome
from tests.cadence.test_reshapedcadence import Test_ReshapedCadence
from tests.cadence.test_reversedcadence import Test_ReversedCadence
from tests.cadence.test_sequence        import Test_Sequence
from tests.cadence.test_tdicadence      import Test_TDICadence

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
