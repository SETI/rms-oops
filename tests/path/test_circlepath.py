################################################################################
# tests/path/test_circlepath.py
################################################################################

import numpy as np
import unittest

from oops      import Frame
from oops.path import Path, CirclePath


class Test_CirclePath(unittest.TestCase):

    def runTest(self):

        np.random.seed(2787)

        # Note: Unit testing is performed in surface/orbitplane.py

        ####################################
        # __getstate__/__setstate__

        radius = 100000.
        lon = 5 * np.random.randn()
        rate = 0.001 * np.random.randn()
        epoch = 10. * 365. * 86400. * np.random.randn()
        origin = Path.SSB
        frame = Frame.J2000
        path = CirclePath(radius, lon, rate, epoch, origin, frame)
        state = path.__getstate__()

        copied = Path.__new__(CirclePath)
        copied.__setstate__(state)
        self.assertEqual(copied.__getstate__(), state)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
