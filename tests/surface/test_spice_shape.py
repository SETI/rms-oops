################################################################################
# tests/surface/test_spice_shape.py
################################################################################

import os
import unittest

import cspyce

from oops.frame.frame_        import Frame
from oops.frame.spiceframe    import SpiceFrame
from oops.path.path_          import Path
from oops.path.spicepath      import SpicePath
from oops.surface.spice_shape import spice_shape
from oops.unittester_support  import TEST_SPICE_PREFIX
import oops.spice_support as spice


class Test_spice_shape(unittest.TestCase):

    def setUp(self):
        spice.initialize()
        paths = TEST_SPICE_PREFIX.retrieve(["pck00010.tpc",
                                            "de421.bsp"])
        for path in paths:
            cspyce.furnsh(path)

    def tearDown(self):
        pass

    def runTest(self):

        _ = SpicePath("VENUS", "SSB", "J2000", path_id="APHRODITE")

        body = spice_shape("VENUS")
        self.assertEqual(Path.as_path(body.origin).path_id, "APHRODITE")
        self.assertEqual(body.req, 6051.8)
        self.assertEqual(body.squash_z, 1.)

################################################################################
