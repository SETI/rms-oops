################################################################################
# tests/hosts/galileo/ssi/__init__.py
################################################################################
import unittest
import os.path
import oops.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


#===============================================================================
# class Test_AAA_Galileo_SSI_index_file(unittest.TestCase):
#
#     #===========================================================================
#     def runTest(self):
#         dir = '/home/spitale/SETI/RMS/metadata/GO_0xxx/GO_0017'
# #        dir = os.path.join(TESTDATA_PARENT_DIRECTORY, 'galileo/GO_0017')
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

############################################

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
