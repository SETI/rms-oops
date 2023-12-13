################################################################################
# oops/unittester.py: Global unit-tester
################################################################################

import oops

from oops.cadence.unittester     import *
from oops.calibration.unittester import *
from oops.fov.unittester         import *
from oops.frame.unittester       import *
from oops.observation.unittester import *
from oops.path.unittester        import *
from oops.surface.unittester     import *
from oops.body        import Test_Body
from oops.event       import Test_Event
from oops.lightsource import Test_LightSource
from oops.transform   import Test_Transform

# from oops.hosts.unittester import *

################################################################################
# To run all unittests...
# python oops/unittester.py

import unittest

if __name__ == '__main__':

    unittest.main(verbosity=2)

################################################################################
