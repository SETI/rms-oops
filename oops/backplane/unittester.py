################################################################################
# oops/backplane/unittester.py
################################################################################

import os
import unittest
import oops.backplane.gold_master as gm
from oops.unittester_support import OOPS_TEST_DATA_PATH

class Test_Backplane_via_gold_master(unittest.TestCase):

  def runTest(self):

    # The d/dv numerical ring derivatives are extra-uncertain due to the high
    # foreshortening in the vertical direction.

    gm.override('SATURN longitude d/du self-check (deg/pix)', 0.3)
    gm.override('SATURN longitude d/dv self-check (deg/pix)', 0.05)
    gm.override('SATURN_MAIN_RINGS azimuth d/dv self-check (deg/pix)', 1.)
    gm.override('SATURN_MAIN_RINGS distance d/dv self-check (km/pix)', 0.3)
    gm.override('SATURN_MAIN_RINGS longitude d/dv self-check (deg/pix)', 1.)
    gm.override('SATURN:RING azimuth d/dv self-check (deg/pix)', 0.1)
    gm.override('SATURN:RING distance d/dv self-check (km/pix)', 0.3)
    gm.override('SATURN:RING longitude d/dv self-check (deg/pix)', 0.1)

    gm.execute_as_unittest(self,
                obspath = os.path.join(OOPS_TEST_DATA_PATH,
                                       'cassini/ISS/W1573721822_1.IMG'),
                module  = 'hosts.cassini.iss',
                planet  = 'SATURN',
                moon    = 'EPIMETHEUS',
                ring    = 'SATURN_MAIN_RINGS')

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
