################################################################################
# oops_/calib/unittester.py
################################################################################

import unittest

from oops_.calib.calibration_ import Test_Calibration
from oops_.calib.extended     import Test_ExtendedSource
from oops_.calib.point        import Test_PointSource

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
