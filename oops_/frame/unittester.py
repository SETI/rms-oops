################################################################################
# oops_/format/unittester.py
################################################################################

import unittest
import cspice

from oops_.frame.frame_         import Test_Frame
from oops_.frame.cmatrix        import Test_Cmatrix
from oops_.frame.inclinedframe  import Test_InclinedFrame
from oops_.frame.ringframe      import Test_RingFrame
from oops_.frame.spiceframe     import Test_SpiceFrame
from oops_.frame.spinframe      import Test_SpinFrame
from oops_.frame.tracker        import Test_Tracker

cspice.furnsh("test_data/spice/naif0009.tls")
cspice.furnsh("test_data/spice/pck00010.tpc")
cspice.furnsh("test_data/spice/de421.bsp")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
