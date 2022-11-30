################################################################################
# hosts/unittester.py
################################################################################

import unittest

from hosts.juno.unittester         import *
#from hosts.cassini.unittester     import *
#from hosts.hst.unittester         import *
#from hosts.voyager.unittester     import *
#from hosts.newhorizons.unittester import *
#from hosts.keck.unittester        import *

########################################
from oops.backplane.unittester_support import backplane_unittester_args

if __name__ == '__main__':
    backplane_unittester_args()
    unittest.main(verbosity=2)
################################################################################
