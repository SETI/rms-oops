################################################################################
# oops/fov/unittester.py
################################################################################

import unittest

from oops.fov               import Test_FOV
from oops.fov.flatfov       import Test_FlatFOV
from oops.fov.barrelfov     import Test_BarrelFOV
from oops.fov.offsetfov     import Test_OffsetFOV
from oops.fov.polyfov       import Test_PolyFOV
from oops.fov.polynomialfov import Test_PolynomialFOV
from oops.fov.radialfov     import Test_RadialFOV
from oops.fov.slicefov      import Test_SliceFOV
from oops.fov.subarray      import Test_Subarray
from oops.fov.subsampledfov import Test_SubsampledFOV
from oops.fov.tdifov        import Test_TDIFOV
from oops.fov.wcsfov        import Test_WCSFOV

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
