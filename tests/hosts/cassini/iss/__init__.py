################################################################################
# tests/hosts/cassini/iss/__init__.py
################################################################################

import unittest
import os.path
import oops.gold_master as gm

from oops.body import Body
from oops.unittester_support import TEST_DATA_PREFIX


class Test_Cassini_ISS_GoldMaster(unittest.TestCase):

    def setUp(self):
        from tests.hosts.cassini.iss import standard_obs

    def test_W1573721822(self):
        gm.execute_as_unittest(self, 'W1573721822_1')

    def tearDown(self):
        Body._undefine_solar_system()
        Body.define_solar_system()

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
