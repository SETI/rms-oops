################################################################################
# oops/format_/unittester.py
################################################################################

import unittest
import cspice
import os.path

from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY
 
from oops.frame_.frame          import Test_Frame
from oops.frame_.cmatrix        import Test_Cmatrix
from oops.frame_.inclinedframe  import Test_InclinedFrame
from oops.frame_.postarg        import Test_PosTarg
from oops.frame_.ringframe      import Test_RingFrame
from oops.frame_.spiceframe     import Test_SpiceFrame
from oops.frame_.spinframe      import Test_SpinFrame
from oops.frame_.tracker        import Test_Tracker

cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "p:/SETI/devel/src/pds-tools/test_data/spice/naif0009.tls"))
cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "p:/SETI/devel/src/pds-tools/test_data/spice/pck00010.tpc"))
cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "p:/SETI/devel/src/pds-tools/test_data/spice/de421.bsp"))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
