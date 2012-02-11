################################################################################
# oops/format/unittester.py
################################################################################

import unittest
import cspice

from oops.frame.baseclass  import Test_Frame

from oops.frame.cmatrix    import Test_Cmatrix
from oops.frame.ringframe  import Test_RingFrame
from oops.frame.spiceframe import Test_SpiceFrame
from oops.frame.spinframe  import Test_SpinFrame

cspice.furnsh("test_data/spice/naif0009.tls")
cspice.furnsh("test_data/spice/pck00010.tpc")
cspice.furnsh("test_data/spice/de421.bsp")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################