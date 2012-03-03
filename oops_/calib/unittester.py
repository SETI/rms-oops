################################################################################
# oops_/calib/unittester.py
################################################################################

import unittest

from calibration_ import Test_Calibration
from distorted    import Test_Distorted
from scaling      import Test_Scaling

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
