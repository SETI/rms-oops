################################################################################
# oops/cadence/unittester.py
################################################################################

import unittest

from hosts.juno.junocam     import Test_Juno_Junocam_Backplane_Exercises
from hosts.juno.jiram.img   import Test_Juno_JIRAM_IMG_Backplane_Exercises
from hosts.juno.jiram.spe   import Test_Juno_JIRAM_SPE_Backplane_Exercises

########################################
from oops.backplane.unittester_support import backplane_unittester_args

if __name__ == '__main__':
    backplane_unittester_args()
    unittest.main(verbosity=2)
################################################################################
