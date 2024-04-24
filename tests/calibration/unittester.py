################################################################################
# oops/calibration/unittester.py
################################################################################

import unittest

from tests.calibration.test_flatcalib import Test_FlatCalib
from tests.calibration.test_radiance  import Test_Radiance
from tests.calibration.test_rawcounts import Test_RawCounts

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
