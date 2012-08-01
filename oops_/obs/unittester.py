################################################################################
# oops_/obs/unittester.py
################################################################################

import unittest

from observation_ import Test_Observation
from pixel        import Test_Pixel
from pushbroom    import Test_Pushbroom
from rasterscan   import Test_RasterScan
from rasterslit   import Test_RasterSlit
from rasterslit1d import Test_RasterSlit1D
from slit         import Test_Slit
from slit1d       import Test_Slit1D
from snapshot     import Test_Snapshot

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
