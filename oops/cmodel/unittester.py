################################################################################
# oops/coord/unittester.py
################################################################################

import unittest

from baseclass  import Test_CoordinateModel
from distance   import Test_Distance
from latitude   import Test_Latitude
from longitude  import Test_Longitude
from radius     import Test_Radius

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
