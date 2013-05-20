################################################################################
# oops_/unittester.py: Global unit-tester
################################################################################

import oops

from oops_.array.unittester   import *
from oops_.cadence.unittester import *
from oops_.calib.unittester   import *
from oops_.cmodel.unittester  import *
from oops_.format.unittester  import *
from oops_.fov.unittester     import *
from oops_.frame.unittester   import *
from oops_.nav.unittester     import *
from oops_.path.unittester    import *
from oops_.surface.unittester import *

from oops_.backplane  import Test_Backplane
from oops_.body       import Test_Body
from oops_.event      import Test_Event
from oops_.transform  import Test_Transform
from oops_.units      import Test_Units

from obs.unittester           import *

################################################################################
# To run all unittests...
# python oops/unittester.py

import unittest

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
