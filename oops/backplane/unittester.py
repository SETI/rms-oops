################################################################################
# oops/backplane/unittester.py
################################################################################
import unittest

from oops.backplane            import Test_Backplane_via_gold_master
from oops.backplane            import Test_Backplane_Surfaces
from oops.backplane            import Test_Backplane_Borders
from oops.backplane            import Test_Backplane_Empty_Events
from oops.backplane.ansa       import Test_Ansa
from oops.backplane.border     import Test_Border
from oops.backplane.distance   import Test_Distance
from oops.backplane.lighting   import Test_Lighting
from oops.backplane.limb       import Test_Limb
from oops.backplane.orbit      import Test_Orbit
from oops.backplane.pole       import Test_Pole
from oops.backplane.resolution import Test_Resolution
from oops.backplane.ring       import Test_Ring
from oops.backplane.sky        import Test_Sky
from oops.backplane.spheroid   import Test_Spheroid
from oops.backplane.where      import Test_Where

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
