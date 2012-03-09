################################################################################
# oops_/calib/unittester.py
################################################################################

import unittest

from oops_.calib.calibration_ import Test_Calibration
from oops_.calib.distorted    import Test_Distorted
from oops_.calib.scaling      import Test_Scaling

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
