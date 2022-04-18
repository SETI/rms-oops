################################################################################
# oops/fov_/unittester.py
################################################################################

import unittest

from oops.fov_.fov        import Test_FOV
from oops.fov_.flatfov    import Test_Flat
from oops.fov_.offsetfov  import Test_OffsetFOV
from oops.fov_.polynomial import Test_Polynomial
#from oops.fov_.radial     import Test_Radial
from oops.fov_.slicefov   import Test_SliceFOV
from oops.fov_.subarray   import Test_Subarray
from oops.fov_.subsampled import Test_Subsampled

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
