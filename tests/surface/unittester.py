################################################################################
# tests/surface/unittester.py
################################################################################

import unittest

from tests.surface.test_ansa             import Test_Ansa
from tests.surface.test_centricellipsoid import Test_CentricEllipsoid
from tests.surface.test_centricspheroid  import Test_CentricSpheroid
from tests.surface.test_ellipsoid        import Test_Ellipsoid
from tests.surface.test_graphicellipsoid import Test_GraphicEllipsoid
from tests.surface.test_graphicspheroid  import Test_GraphicSpheroid
from tests.surface.test_limb             import Test_Limb
from tests.surface.test_orbitplane       import Test_OrbitPlane
# from tests.surface.test_polarlimb        import Test_PolarLimb NOT WORKING!
from tests.surface.test_ringplane        import Test_RingPlane
from tests.surface.test_spheroid         import Test_Spheroid
from tests.surface.test_spice_shape      import Test_spice_shape

########################################
if __name__ == '__main__':

    import oops
    oops.config.LOGGING.on('     ')

    unittest.main(verbosity=2)
################################################################################
