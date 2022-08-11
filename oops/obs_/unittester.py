################################################################################
# oops/obs_/unittester.py
################################################################################

import unittest

from oops.obs_.observation  import Test_Observation
from oops.obs_.insitu       import Test_InSitu
from oops.obs_.pixel        import Test_Pixel
from oops.obs_.pushframe    import Test_Pushframe
from oops.obs_.pushbroom    import Test_Pushbroom
from oops.obs_.rasterscan   import Test_RasterScan
from oops.obs_.rasterslit   import Test_RasterSlit
from oops.obs_.rasterslit1d import Test_RasterSlit1D
from oops.obs_.slit         import Test_Slit
from oops.obs_.slit1d       import Test_Slit1D
from oops.obs_.snapshot     import Test_Snapshot

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
