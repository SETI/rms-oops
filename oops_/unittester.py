################################################################################
# oops_/unittester.py: Global unit-tester
################################################################################

from array_.unittester  import *
from calib.unittester   import *
from cmodel.unittester  import *
from format.unittester  import *
from fov.unittester     import *
from frame.unittester   import *
#from inst.unittester    import *
#from obs.unittester     import *
from path.unittester    import *
from surface.unittester import *

from body       import Test_Body
from event      import Test_Event
from transform  import Test_Transform
from units      import Test_Units

################################################################################
# To run all unittests...
# python oops/unittester.py

import unittest

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
