################################################################################
# oops/calibration/unittester.py
################################################################################

import unittest

from oops.calibration                import Test_Calibration
from oops.calibration.extendedsource import Test_ExtendedSource
from oops.calibration.pointsource    import Test_PointSource

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
