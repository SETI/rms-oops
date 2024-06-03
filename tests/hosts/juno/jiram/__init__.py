################################################################################
# tests/hosts/juno/jiram/__init__.py
################################################################################

import unittest
import os.path
import oops.backplane.gold_master as gm
import oops.hosts.juno.jiram as jiram

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


#===============================================================================
class Test_Juno_JIRAM_GoldMaster(unittest.TestCase):

    #===========================================================================
    def setUp(self):
        gm.define_standard_obs('JIR_IMG_RDR_2013282T133843_V03',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2000/DATA/'
                                       'JIR_IMG_RDR_2013282T133843_V03.IMG'),
                index   = 1,
                module  = 'oops.hosts.juno.jiram',
                planet  = '',
                moon    = 'MOON',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})

        gm.define_standard_obs('JIR_IMG_RDR_2017244T104633_V01',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2008/DATA/'
                                       'JIR_IMG_RDR_2017244T104633_V01.IMG'),
                index   = 1,
                module  = 'oops.hosts.juno.jiram',
                planet  = '',
                moon    = 'EUROPA',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})

        gm.define_standard_obs('JIR_IMG_RDR_2018197T055537_V01',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2014/DATA/'
                                       'JIR_IMG_RDR_2018197T055537_V01.IMG'),
                index   = 0,
                module  = 'oops.hosts.juno.jiram',
                planet  = 'JUPITER',
                moon    = '',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})

        gm.define_standard_obs('JIR_SPE_RDR_2013282T133845_V03',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2000/DATA/'
                                       'JIR_SPE_RDR_2013282T133845_V03.DAT'),
                index   = 0,
                module  = 'oops.hosts.juno.jiram',
                planet  = '',
                moon    = 'MOON',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})


    #===========================================================================
    def test_1(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_IMG_RDR_2013282T133843_V03')

    #===========================================================================
    def test_2(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_IMG_RDR_2017244T104633_V01')

    #===========================================================================
    def test_3(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_IMG_RDR_2018197T055537_V01')

    #===========================================================================
    def test_4(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_SPE_RDR_2013282T133845_V03')


##############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
