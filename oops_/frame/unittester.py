################################################################################
# oops_/format/unittester.py
################################################################################

import unittest
import cspice

from frame_     import Test_Frame

from cmatrix    import Test_Cmatrix
from ringframe  import Test_RingFrame
from spiceframe import Test_SpiceFrame
from spinframe  import Test_SpinFrame

cspice.furnsh("test_data/spice/naif0009.tls")
cspice.furnsh("test_data/spice/pck00010.tpc")
cspice.furnsh("test_data/spice/de421.bsp")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
