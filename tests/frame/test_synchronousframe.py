################################################################################
# tests/frame/test_synchronousframe.py
################################################################################

import numpy as np
import unittest

from polymath   import Scalar
from oops       import Body
from oops.frame import SynchronousFrame


class Test_SynchronousFrame(unittest.TestCase):

    def setUp(self):
        Body.reset_registry()
        Body.define_solar_system('2000-01-01', '2020-01-01')

    def tearDown(self):
        Body._undefine_solar_system()
        Body.define_solar_system()

    def runTest(self):
        from oops.path import Path

        # Path of Saturn relative to Enceladus
        inward = Path.as_path('SATURN').wrt('ENCELADUS')
        synchro = SynchronousFrame('ENCELADUS', 'SATURN')

        time = Scalar(np.arange(1000.) * 86400.)

        # Make sure direction to Saturn is along X-axis
        pos = inward.event_at_time(time).wrt_frame(synchro).pos
        ### self.assertTrue(np.all(pos.vals[:,0] > 0.))
        self.assertLess(np.max(np.abs(pos.vals[:,1])), 1.e-10)
        self.assertLess(np.max(np.abs(pos.vals[:,2])), 1.e-10)

        # Make sure this frame and IAU_ENCELADUS are close
        xform = synchro.wrt('IAU_ENCELADUS').transform_at_time(time)

        self.assertLess(np.max(np.abs(xform.omega.vals[:,0])), 5.e-8)
        self.assertLess(np.max(np.abs(xform.omega.vals[:,1])), 5.e-8)
        self.assertLess(np.max(np.abs(xform.omega.vals[:,2])), 1.e-6)

        unit = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.assertLess(np.median(np.abs(xform.matrix.vals - unit).ravel()), 5.e-4)
        self.assertLess(np.median(np.abs(xform.matrix.vals - unit).ravel()), 0.1)

################################################################################
