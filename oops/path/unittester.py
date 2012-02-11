################################################################################
# oops/path/unittester.py
################################################################################

import unittest
import cspice

from oops.path.baseclass  import Test_Path
from oops.path.multipath  import Test_MultiPath
from oops.path.spicepath  import Test_SpicePath

cspice.furnsh("test_data/spice/naif0009.tls")
cspice.furnsh("test_data/spice/pck00010.tpc")
cspice.furnsh("test_data/spice/de421.bsp")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
