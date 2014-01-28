################################################################################
# oops/array_/unittester.py
################################################################################

import unittest

from oops.array_.utils     import Test_utils

from oops.array_.array     import Test_Array
from oops.array_.empty     import Test_Empty
from oops.array_.matrix3   import Test_Matrix3
from oops.array_.matrixn   import Test_MatrixN
from oops.array_.pair      import Test_Pair
from oops.array_.scalar    import Test_Scalar
from oops.array_.tuple     import Test_Tuple
from oops.array_.vector3   import Test_Vector3
from oops.array_.vectorn   import Test_VectorN

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
