################################################################################
# hosts/juno/unittester.py
################################################################################

import unittest

from hosts.juno.junocam   import Test_Juno_Junocam_GoldMaster
# from hosts.juno.junocam   import Test_Juno_Junocam_Backplane_Exercises
# from hosts.juno.jiram.img import Test_Juno_JIRAM_IMG_Backplane_Exercises
# from hosts.juno.jiram.spe import Test_Juno_JIRAM_SPE_Backplane_Exercises

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
