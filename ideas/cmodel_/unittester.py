################################################################################
# oops/cmodel_/coord/unittester.py
################################################################################

import unittest

from oops.cmodel_.cmodel     import Test_CoordinateModel
from oops.cmodel_.distance   import Test_Distance
from oops.cmodel_.latitude   import Test_Latitude
from oops.cmodel_.longitude  import Test_Longitude
from oops.cmodel_.radius     import Test_Radius

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
