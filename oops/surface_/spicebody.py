################################################################################
# oops_/surface/spicebody.py: Handles bodys shaped defined in the SPICE toolkit.
#
# Note: This is not a Surface subclass. Instead, it is a function to return a
# surface (either a Spheroid or an Ellipsoid) using information found in the
# SPICE toolkit.
#
# 2/17/12 Created (MRS)
################################################################################

import cspice

from oops_.surface.spheroid import Spheroid
from oops_.surface.ellipsoid import Ellipsoid
from oops_.path.spicepath import SpicePath
from oops_.frame.spiceframe import SpiceFrame
import oops_.registry as registry
import oops_.spice_support as spice

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

        cspice.furnsh("test_data/spice/naif0009.tls")
        cspice.furnsh("test_data/spice/pck00010.tpc")
        cspice.furnsh("test_data/spice/de421.bsp")

        ignore = SpicePath("VENUS", "SSB", "J2000", "APHRODITE")
        ignore = SpiceFrame("VENUS", "J2000", "SLOWSPINNER")

        body = spice_body("VENUS")
        self.assertEqual(body.origin_id, "APHRODITE")
        self.assertEqual(body.frame_id,  "SLOWSPINNER")
        self.assertEqual(body.req, 6051.8)
        self.assertEqual(body.squash_z, 1.)

        registry.initialize()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
