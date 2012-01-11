import unittest
import cspice

import oops

from utils                  import Test_utils

from broadcastable.Scalar   import Test_Scalar
from broadcastable.Pair     import Test_Pair
from broadcastable.Vector3  import Test_Vector3
from broadcastable.Matrix3  import Test_Matrix3
#from broadcastable.Tuple    import Test_Tuple

from Transform              import Test_Transform
from Event                  import Test_Event

from fov.FOV                import Test_FOV
from fov.FlatFOV            import Test_FlatFOV
from fov.PolynomialFOV      import Test_PolynomialFOV
from fov.SubarrayFOV        import Test_SubarrayFOV
from fov.SubsampledFOV      import Test_SubsampledFOV

from frame.Frame            import Test_Frame
from frame.Cmatrix          import Test_Cmatrix
from frame.MatrixFrame      import Test_MatrixFrame
from frame.RingFrame        import Test_RingFrame
from frame.SpiceFrame       import Test_SpiceFrame

from path.Path              import Test_Path
from path.MultiPath         import Test_MultiPath
from path.SpicePath         import Test_SpicePath

from surface.Surface        import Test_Surface
from surface.RingPlane      import Test_RingPlane

from observation.Observation import Test_Observation

# Set CSPICE up for testing...

cspice.furnsh("test_data/spice/naif0009.tls")
cspice.furnsh("test_data/spice/pck00010.tpc")
cspice.furnsh("test_data/spice/de421.bsp")

################################################################################
# To run all unittests...
# python oops/unittester.py

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
