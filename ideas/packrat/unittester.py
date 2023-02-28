################################################################################
# packrat/unittester.py: Global unit-tester
################################################################################

from packrat import Test_Packrat
from packrat_arrays import Test_Packrat_arrays

################################################################################
# To run all unittests...
# python packrat/unittester.py

import unittest

if __name__ == '__main__': # pragma: no cover

    unittest.main(verbosity=2)

################################################################################
