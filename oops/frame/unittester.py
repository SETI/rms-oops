################################################################################
# oops/frame/unittester.py
################################################################################

import unittest

from oops.frame                  import Test_Frame
from oops.frame.cmatrix          import Test_Cmatrix
from oops.frame.inclinedframe    import Test_InclinedFrame
from oops.frame.laplaceframe     import Test_LaplaceFrame
from oops.frame.navigation       import Test_Navigation
from oops.frame.poleframe        import Test_PoleFrame
from oops.frame.postargframe     import Test_PosTargFrame
from oops.frame.ringframe        import Test_RingFrame
from oops.frame.rotation         import Test_Rotation
from oops.frame.spiceframe       import Test_SpiceFrame
from oops.frame.spicetype1frame  import Test_SpiceType1Frame
from oops.frame.spinframe        import Test_SpinFrame
from oops.frame.synchronousframe import Test_SynchronousFrame
from oops.frame.trackerframe     import Test_TrackerFrame
from oops.frame.twovectorframe   import Test_TwoVectorFrame

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
