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
from hosts.solar.unittester        import *

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
