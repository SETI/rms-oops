################################################################################
# oops_/fov/unittester.py
################################################################################

import unittest

from baseclass  import Test_FOV
from flat       import Test_Flat
from polynomial import Test_Polynomial
from subarray   import Test_Subarray
from subsampled import Test_Subsampled

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
