################################################################################
# oops/path_/unittester.py
################################################################################

import unittest

from oops.path_.path       import Test_Path
from oops.path_.circlepath import Test_CirclePath
from oops.path_.coordpath  import Test_CoordPath
from oops.path_.fixedpath  import Test_FixedPath
#from oops.path_.kepler     import Test_Kepler 
from oops.path_.linearpath import Test_LinearPath
from oops.path_.multipath  import Test_MultiPath
from oops.path_.spicepath  import Test_SpicePath

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
