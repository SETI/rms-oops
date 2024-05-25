################################################################################
# tests/frame/test_postargframe.py
################################################################################

import unittest

from polymath   import Vector3
from oops.frame import Frame, PosTargFrame


class Test_PosTargFrame(unittest.TestCase):

    def runTest(self):

#         Frame.reset_registry()

        postarg = PosTargFrame(0.0001, 0.0002, "J2000")
        transform = postarg.transform_at_time(0.)
        rotated = transform.rotate(Vector3.ZAXIS)

        self.assertTrue(abs(rotated.vals[0] - 0.0001) < 1.e-8)
        self.assertTrue(abs(rotated.vals[1] - 0.0002) < 1.e-8)

#         Frame.reset_registry()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
