###############################################################################
# oops/cadence_/unittester.py
################################################################################

import unittest

from oops.cadence_.cadence   import Test_Cadence
from oops.cadence_.dual      import Test_DualCadence
from oops.cadence_.metronome import Test_Metronome
from oops.cadence_.reshaped  import Test_ReshapedCadence
from oops.cadence_.sequence  import Test_Sequence

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
