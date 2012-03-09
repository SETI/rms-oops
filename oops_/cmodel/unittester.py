################################################################################
# oops_/coord/unittester.py
################################################################################

import unittest

from oops_.cmodel.cmodel_    import Test_CoordinateModel
from oops_.cmodel.distance   import Test_Distance
from oops_.cmodel.latitude   import Test_Latitude
from oops_.cmodel.longitude  import Test_Longitude
from oops_.cmodel.radius     import Test_Radius

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
