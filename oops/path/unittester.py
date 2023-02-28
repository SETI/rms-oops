################################################################################
# oops/path/unittester.py
################################################################################

import unittest

from oops.path            import Test_Path
from oops.path.circlepath import Test_CirclePath
from oops.path.coordpath  import Test_CoordPath
from oops.path.fixedpath  import Test_FixedPath
from oops.path.keplerpath import Test_KeplerPath
from oops.path.linearpath import Test_LinearPath
from oops.path.multipath  import Test_MultiPath
from oops.path.spicepath  import Test_SpicePath

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
