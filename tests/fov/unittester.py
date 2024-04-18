################################################################################
# oops/fov/unittester.py
################################################################################

import unittest

from tests.fov.test_barrelfov     import Test_BarrelFOV
from tests.fov.test_flatfov       import Test_FlatFOV
from tests.fov.test_polynomialfov import Test_PolynomialFOV
from tests.fov.test_subarray      import Test_Subarray
from tests.fov.test_subsampledfov import Test_SubsampledFOV
from tests.fov.test_tdifov        import Test_TDIFOV
from tests.fov.test_wcsfov        import Test_WCSFOV

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
