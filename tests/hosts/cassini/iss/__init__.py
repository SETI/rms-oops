################################################################################
# tests/hosts/cassini/iss/__init__.py
################################################################################

import unittest
import os.path
import oops.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


class Test_Cassini_ISS_GoldMaster(unittest.TestCase):

    def setUp(self):
        from tests.hosts.cassini.iss import standard_obs

    def test_W1573721822(self):
        gm.execute_as_unittest(self, 'W1573721822_1')

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
