################################################################################
# oops_/array_/unittester.py
################################################################################

import unittest

from oops_.array.utils     import Test_utils

from oops_.array.array_    import Test_Array
from oops_.array.empty     import Test_Empty
from oops_.array.matrix3   import Test_Matrix3
from oops_.array.matrixn   import Test_MatrixN
from oops_.array.pair      import Test_Pair
from oops_.array.scalar    import Test_Scalar
from oops_.array.tuple     import Test_Tuple
from oops_.array.vector3   import Test_Vector3
from oops_.array.vectorn   import Test_VectorN

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
