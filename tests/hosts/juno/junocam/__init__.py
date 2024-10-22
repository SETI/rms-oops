################################################################################
# oops/inst/juno/junocam.py
################################################################################

import os
import unittest
import oops.backplane.gold_master as gm

from oops.unittester_support import TEST_DATA_PREFIX

#===============================================================================
class Test_Juno_Junocam_GoldMaster(unittest.TestCase):

    #===========================================================================
    def setUp(self):
        from oops.hosts.juno.junocam import standard_obs

    #===========================================================================
    def test_JNCR_2016347_03C00192_V01(self):
        gm.execute_as_unittest(self, 'JNCR_2016347_03C00192_V01')

    #===========================================================================
    def test_JNCR_2020366_31C00065_V01(self):
        gm.execute_as_unittest(self, 'JNCR_2020366_31C00065_V01')

    #===========================================================================
    def test_JNCR_2019096_19M00012_V02(self):
        gm.execute_as_unittest(self, 'JNCR_2019096_19M00012_V02')

    #===========================================================================
    def test_JNCR_2019149_20G00008_V01(self):
        gm.execute_as_unittest(self, 'JNCR_2019149_20G00008_V01')



##############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
