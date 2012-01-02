import numpy as np
import pylab
import unittest

import oops

################################################################################
# Snapshot Class
################################################################################

class Snapshot(oops.Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time."""

    def __init__(self, data, mask, axes, time, fov, path_id, frame_id,
                       calibration):

        self.data = data
        self.mask = mask
        self.path_id = path_id
        self.frame_id = frame_id
        self.t0 = time[0]
        self.t1 = time[1]
        self.fov = fov
        self.calibration = calibration

        self.u_axis = axes.index("u")
        self.v_axis = axes.index("v")
        self.t_axis = None

########################################
# UNIT TESTS
########################################

class Test_Snapshot(unittest.TestCase):

    def runTest(self):

        pass

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
