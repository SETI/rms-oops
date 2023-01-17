################################################################################
# pds-oops/unittester.py: Global unit-tester for oops + applications
################################################################################
from hosts.unittester import *
from oops.unittester  import *

################################################################################
# To run all unittests...
# python oops/unittester.py

import unittest

if __name__ == '__main__':

    unittest.main(verbosity=2)

################################################################################
