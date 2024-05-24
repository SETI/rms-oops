################################################################################
# oops/hosts/unittester.py
################################################################################

import unittest

#from tests.hosts.juno         import *
from tests.hosts.galileo.ssi     import *
from tests.hosts.cassini.iss     import *
#from tests.hosts.hst         import *
#from tests.hosts.voyager     import *
#from tests.hosts.newhorizons import *
#from tests.hosts.keck        import *

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
