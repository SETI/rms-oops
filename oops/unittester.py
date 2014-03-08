################################################################################
# oops/unittester.py: Global unit-tester
################################################################################

import oops

from oops.cadence_.unittester import *
from oops.calib_.unittester   import *
from oops.cmodel_.unittester  import *
from oops.format_.unittester  import *
from oops.fov_.unittester     import *
from oops.frame_.unittester   import *
from oops.obs_.unittester     import *
from oops.path_.unittester    import *
from oops.surface_.unittester import *

from oops.backplane  import Test_Backplane
from oops.body       import Test_Body
from oops.event      import Test_Event
from oops.transform  import Test_Transform

################################################################################
# To run all unittests...
# python oops/unittester.py

import unittest

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
