################################################################################
# oops_/cadence/unittester.py
################################################################################

import unittest

from oops_.cadence.cadence_  import Test_Cadence
from oops_.cadence.dual      import Test_DualCadence
from oops_.cadence.metronome import Test_Metronome
from oops_.cadence.reshaped  import Test_ReshapedCadence
from oops_.cadence.sequence  import Test_Sequence

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
