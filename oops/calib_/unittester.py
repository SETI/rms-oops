################################################################################
# oops/calib_/unittester.py
################################################################################

import unittest

from oops.calib_.calibration import Test_Calibration
from oops.calib_.extended    import Test_ExtendedSource
from oops.calib_.point       import Test_PointSource

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
