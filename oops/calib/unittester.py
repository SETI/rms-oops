################################################################################
# oops/calib/unittester.py
################################################################################

import unittest

from oops.calib.baseclass  import Test_Calibration
from oops.calib.distorted  import Test_Distorted
from oops.calib.scaling    import Test_Scaling

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
