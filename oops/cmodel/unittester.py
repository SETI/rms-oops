################################################################################
# oops/coord/unittester.py
################################################################################

import unittest

from oops.cmodel.baseclass  import Test_CoordinateModel
from oops.cmodel.distance   import Test_Distance
from oops.cmodel.latitude   import Test_Latitude
from oops.cmodel.longitude  import Test_Longitude
from oops.cmodel.radius     import Test_Radius

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
