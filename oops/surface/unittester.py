################################################################################
# oops/surface/unittester.py
################################################################################

import unittest

from oops.surface                  import Test_Surface
from oops.surface.ansa             import Test_Ansa
from oops.surface.centricellipsoid import Test_CentricEllipsoid
from oops.surface.centricspheroid  import Test_CentricSpheroid
from oops.surface.ellipsoid        import Test_Ellipsoid
from oops.surface.graphicellipsoid import Test_GraphicEllipsoid
from oops.surface.graphicspheroid  import Test_GraphicSpheroid
from oops.surface.limb             import Test_Limb
from oops.surface.nullsurface      import Test_NullSurface
from oops.surface.orbitplane       import Test_OrbitPlane
from oops.surface.polarlimb        import Test_PolarLimb
from oops.surface.ringplane        import Test_RingPlane
from oops.surface.spheroid         import Test_Spheroid
from oops.surface.spice_shape      import Test_spice_shape

########################################
if __name__ == '__main__':

    import oops
    oops.config.LOGGING.on('     ')

    unittest.main(verbosity=2)

################################################################################
