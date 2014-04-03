################################################################################
# oops/surface_/spicebody.py: For bodies shaped defined in SPICE.
################################################################################

import cspice

from oops.surface_.spheroid  import Spheroid
from oops.surface_.ellipsoid import Ellipsoid
from oops.path_.spicepath    import SpicePath
from oops.frame_.spiceframe  import SpiceFrame
import oops.spice_support as spice

def spice_body(spice_id):
    """Returns a Spheroid or Ellipsoid defining the path, orientation and shape
    of a body defined in the SPICE toolkit.

    Input:
        spice_id        the name or ID of the body as defined in the SPICE
                        toolkit.
        centric         True to use planetocentric latitudes, False to use
                        planetographic latitudes.
    """

    spice_body_id = spice.body_id_and_name(spice_id)[0]
    origin_id = spice.PATH_TRANSLATION[spice_body_id]

    spice_frame_name = spice.frame_id_and_name(spice_id)[1]
    frame_id = spice.FRAME_TRANSLATION[spice_frame_name]

    radii = cspice.bodvcd(spice_body_id, "RADII")

    if radii[0] == radii[1]:
        return Spheroid(origin_id, frame_id, (radii[0], radii[2]))
    else:
        return Ellipsoid(origin_id, frame_id, radii)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_spice_body(unittest.TestCase):
    
    def runTest(self):

        from oops.path_.path import Path
        from oops.frame_.frame import Frame
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
        import os.path

        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "pck00010.tpc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "de421.bsp"))

        ignore = SpicePath("VENUS", "SSB", "J2000", "APHRODITE")
        ignore = SpiceFrame("VENUS", "J2000", "SLOWSPINNER")

        body = spice_body("VENUS")
        self.assertEqual(body.origin_id, "APHRODITE")
        self.assertEqual(body.frame_id,  "SLOWSPINNER")
        self.assertEqual(body.req, 6051.8)
        self.assertEqual(body.squash_z, 1.)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
