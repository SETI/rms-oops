################################################################################
# tests/observation/unittester.py
################################################################################

import unittest

from tests.observation.test_pixel        import Test_Pixel
from tests.observation.test_rasterslit1d import Test_RasterSlit1D
from tests.observation.test_slit1d       import Test_Slit1D
from tests.observation.test_snapshot     import Test_Snapshot
from tests.observation.test_timedimage   import Test_TimedImage

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
