################################################################################
# oops_/calib/unittester.py
################################################################################

import unittest

from baseclass  import Test_Calibration
from distorted  import Test_Distorted
from scaling    import Test_Scaling

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
