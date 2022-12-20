################################################################################
# oops/observation/unittester.py
################################################################################

import unittest

from oops.observation              import Test_Observation
from oops.observation.insitu       import Test_InSitu
from oops.observation.pixel        import Test_Pixel
from oops.observation.pushframe    import Test_Pushframe    # now TimedImage
from oops.observation.pushbroom    import Test_Pushbroom    # now TimedImage
from oops.observation.rasterscan   import Test_RasterScan   # now TimedImage
from oops.observation.rasterslit   import Test_RasterSlit   # now TimedImage
from oops.observation.rasterslit1d import Test_RasterSlit1D
from oops.observation.slit         import Test_Slit         # now TimedImage
from oops.observation.slit1d       import Test_Slit1D
from oops.observation.snapshot     import Test_Snapshot
from oops.observation.timedimage   import Test_TimedImage

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
