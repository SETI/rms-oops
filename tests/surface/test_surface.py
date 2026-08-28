################################################################################
# tests/surface/test_surface.py: Abstract class Surface
################################################################################

import numpy as np

from polymath      import Vector3
from oops.surface  import Surface


def test_surface():

    np.random.seed(6631)

    # Most methods are heavily tested elsewhere

    # Surface.resolution...

    # Make sure the rotated resolution vectors are perpendicular
    dpos_duv = Vector3(np.random.randn(10000, 3, 2), drank=1)
    (new_du, new_dv) = Surface.resolution(dpos_duv, _unittest=True)
    assert new_du.dot(new_dv).max() < 1.e-12

    # We also expect area to be conserved
    dpos_du = Vector3(dpos_duv.values[...,0])
    dpos_dv = Vector3(dpos_duv.values[...,1])
    diffs = dpos_du.cross(dpos_dv) - new_du.cross(new_dv)
    assert diffs.norm().max() < 1.e-14

################################################################################
