################################################################################
# tests/path/unittester.py
################################################################################

import unittest

from tests.path.test_path       import Test_Path
from tests.path.test_circlepath import Test_CirclePath
from tests.path.test_keplerpath import Test_KeplerPath
from tests.path.test_multipath  import Test_MultiPath
from tests.path.test_spicepath  import Test_SpicePath

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
