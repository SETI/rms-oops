################################################################################
# tests/frame/unittester.py
################################################################################

import unittest

from tests.frame.test_frame            import Test_Frame
from tests.frame.test_cmatrix          import Test_Cmatrix
from tests.frame.test_poleframe        import Test_PoleFrame
from tests.frame.test_postargframe     import Test_PosTargFrame
from tests.frame.test_ringframe        import Test_RingFrame
from tests.frame.test_spiceframe       import Test_SpiceFrame
from tests.frame.test_spinframe        import Test_SpinFrame
from tests.frame.test_synchronousframe import Test_SynchronousFrame
from tests.frame.test_trackerframe     import Test_TrackerFrame

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
