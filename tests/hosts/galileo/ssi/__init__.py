################################################################################
# tests/hosts/galileo/ssi/__init__.py
################################################################################
import unittest
import os.path
import oops.gold_master as gm

from oops.body import Body
from oops.unittester_support import TEST_DATA_PREFIX


#===============================================================================
# class Test_AAA_Galileo_SSI_index_file(unittest.TestCase):
#
#     #===========================================================================
#     def runTest(self):
#         dir = '/home/spitale/SETI/RMS/metadata/GO_0xxx/GO_0017'
# #        dir = f'{OOPS_TEST_DATA_PATH}/galileo/GO_0017'
#
#         obs = from_index(os.path.join(dir, 'GO_0017_index.lbl'),
#                          os.path.join(dir, 'GO_0017_supplemental_index.lbl'))
#
#===============================================================================
class Test_Galileo_SSI_GoldMaster(unittest.TestCase):

    def setUp(self):
        from tests.hosts.galileo.ssi import standard_obs

    #===========================================================================
    def test_C0349632100R(self):
        gm.execute_as_unittest(self, 'C0349632100R')

    #===========================================================================
    def test_C0368369200R(self):
        gm.execute_as_unittest(self, 'C0368369200R')

    #===========================================================================
    def test_C0061455700R(self):
        gm.execute_as_unittest(self, 'C0061455700R')

    #===========================================================================
    def test_C0374685140R(self):
        gm.execute_as_unittest(self, 'C0374685140R')

    #===========================================================================
    def tearDown(self):
        Body._undefine_solar_system()
        Body.define_solar_system()

############################################

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
