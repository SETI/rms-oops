################################################################################
# oops_/fov/unittester.py
################################################################################

import unittest

from oops_.fov.fov_       import Test_FOV
from oops_.fov.flat       import Test_Flat
from oops_.fov.polynomial import Test_Polynomial
from oops_.fov.subarray   import Test_Subarray
from oops_.fov.subsampled import Test_Subsampled

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
