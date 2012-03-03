################################################################################
# oops_/array_/unittester.py
################################################################################

import unittest

from utils     import Test_utils

from baseclass import Test_Array
from empty     import Test_Empty
from matrix3   import Test_Matrix3
from matrixn   import Test_MatrixN
from pair      import Test_Pair
from scalar    import Test_Scalar
from tuple     import Test_Tuple
from vector3   import Test_Vector3
from vectorn   import Test_VectorN

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
