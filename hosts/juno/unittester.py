################################################################################
# oops/cadence/unittester.py
################################################################################

import unittest

from hosts.juno.junocam import Test_Juno_Junocam_GoldMaster_JNCR_2016347_03C00192_V01
from hosts.juno.jiram   import Test_Juno_JIRAM_GoldMaster_JIR_IMG_RDR_2013282T133843_V03
from hosts.juno.jiram   import Test_Juno_JIRAM_GoldMaster_JIR_IMG_RDR_2017244T104633_V01
from hosts.juno.jiram   import Test_Juno_JIRAM_GoldMaster_JIR_IMG_RDR_2018197T055537_V01
from hosts.juno.jiram   import Test_Juno_JIRAM_GoldMaster_JIR_SPE_RDR_2013282T133845_V03

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
