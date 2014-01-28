################################################################################
# oops/path_/unittester.py
################################################################################

import unittest
import cspice
import os.path

from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY

from oops.path_.path       import Test_Path
from oops.path_.circlepath import Test_CirclePath
from oops.path_.fixedpath  import Test_FixedPath
from oops.path_.multipath  import Test_MultiPath
from oops.path_.spicepath  import Test_SpicePath

cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "test_data/spice/naif0009.tls"))
cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "test_data/spice/pck00010.tpc"))
cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "test_data/spice/de421.bsp"))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
