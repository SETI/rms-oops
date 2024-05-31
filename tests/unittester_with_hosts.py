################################################################################
# tests/unittester.py
################################################################################

import unittest

from tests.cadence.unittester     import *
from tests.calibration.unittester import *
from tests.fov.unittester         import *
from tests.frame.unittester       import *
from tests.gravity.unittester     import *
from tests.observation.unittester import *
from tests.path.unittester        import *
from tests.surface.unittester     import *
from tests.test_body              import *
from tests.test_event             import *
from tests.test_transform         import *
from tests.test_utils             import *

from tests.hosts.galileo.ssi      import *
from tests.hosts.cassini.iss      import *

################################################################################
# To run all unittests...
# python oops/unittester.py

if __name__ == '__main__':

    unittest.main(verbosity=2)

################################################################################
