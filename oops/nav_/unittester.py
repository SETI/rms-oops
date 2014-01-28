################################################################################
# oops_/nav/unittester.py
################################################################################

import unittest

from navigation_ import Test_Navigation
from nullnav     import Test_NullNav
from platescale  import Test_PlateScale
from repointing  import Test_Repointing
from timeshift   import Test_TimeShift

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
