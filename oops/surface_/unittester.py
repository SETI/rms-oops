################################################################################
# oops_/surface/unittester.py
################################################################################

import unittest

from surface_   import Test_Surface
from ellipsoid  import Test_Ellipsoid
from limb       import Test_Limb
from orbitplane import Test_OrbitPlane
from ringplane  import Test_RingPlane
from spheroid   import Test_Spheroid

from spicebody import Test_spice_body

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
