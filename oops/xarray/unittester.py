################################################################################
# oops/array/unittester.py
################################################################################

import unittest

from oops.xarray.utils     import Test_utils

from oops.xarray.baseclass import Test_Array
from oops.xarray.empty     import Test_Empty
from oops.xarray.matrix3   import Test_Matrix3
from oops.xarray.pair      import Test_Pair
from oops.xarray.scalar    import Test_Scalar
from oops.xarray.tuple     import Test_Tuple
from oops.xarray.vector3   import Test_Vector3

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
