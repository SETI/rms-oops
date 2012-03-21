################################################################################
# oops_/path/unittester.py
################################################################################

import unittest
import cspice

from path_      import Test_Path
from circlepath import Test_CirclePath
from multipath  import Test_MultiPath
from spicepath  import Test_SpicePath

cspice.furnsh("test_data/spice/naif0009.tls")
cspice.furnsh("test_data/spice/pck00010.tpc")
cspice.furnsh("test_data/spice/de421.bsp")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
