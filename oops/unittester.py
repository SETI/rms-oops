################################################################################
# oop/unittester.py: Global unit-tester
################################################################################

from xarray.unittester  import *

from calib.unittester   import *
from cmodel.unittester  import *
from format.unittester  import *
from fov.unittester     import *
from frame.unittester   import *
#from inst.unittester    import *
#from obs.unittester     import *
from path.unittester    import *
from surface.unittester import *

from oops.body          import Test_Body
from oops.event         import Test_Event
from oops.transform     import Test_Transform
from oops.units         import Test_Units

################################################################################
# To run all unittests...
# python oops/unittester.py

import unittest

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
