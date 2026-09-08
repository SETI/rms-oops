##########################################################################################
# oops/fov/platescale.py: Platescale subclass of class FOV
##########################################################################################

import numpy as np
import pytest

from polymath import Pair
from oops.fov import FlatFOV, Platescale


def test_platescale():
    np.random.seed(1170)

    FACTOR = 2.
    flat = FlatFOV((1.e-4, 1.2e-4), (60, 40))
    scaled = Platescale(FACTOR, flat)

    assert scaled.nparams == 1
    assert scaled.params == (FACTOR,)

    # The pixels cover the same range of (u,v), but each subtends more sky
    assert scaled.uv_shape == flat.uv_shape
    assert scaled.uv_los == flat.uv_los
    assert scaled.uv_scale == flat.uv_scale * FACTOR

    # (x,y) scale with the factor, so the area of a pixel scales with its square
    assert scaled.uv_area == flat.uv_area * FACTOR**2

    uv = Pair(np.random.rand(200, 2) * np.array([60., 40.]))
    assert scaled.xy_from_uvt(uv) == flat.xy_from_uvt(uv) * FACTOR

    # ...which keeps the area of a pixel at its nominal value. Were uv_area not
    # scaled, this factor would come out at the square of the plate scale instead
    assert np.all(np.abs(scaled.area_factor(uv).vals - 1.) < 1.e-3)
    assert np.all(np.abs(flat.area_factor(uv).vals - 1.) < 1.e-3)

    # (u,v) and (x,y) still convert back and forth
    assert np.all(np.abs(scaled.uv_from_xyt(scaled.xy_from_uvt(uv)).vals - uv.vals)
                  < 1.e-12)

    # A factor of one leaves the FOV alone
    unscaled = Platescale(1., flat)
    assert unscaled.uv_scale == flat.uv_scale
    assert unscaled.uv_area == flat.uv_area
    assert unscaled.xy_from_uvt(uv) == flat.xy_from_uvt(uv)

    # Refitting the scale factor updates everything that depends on it
    scaled.set_params(np.array([3.]))
    assert scaled.factor == 3.
    assert scaled.params == (3.,)
    assert scaled.uv_scale == flat.uv_scale * 3.
    assert scaled.uv_area == flat.uv_area * 9.
    assert scaled.xy_from_uvt(uv) == flat.xy_from_uvt(uv) * 3.

    # The wrong number of parameters is rejected
    with pytest.raises(ValueError, match='requires 1 fit'):
        scaled.set_params(np.array([1., 2.]))

    # Freezing prevents any further refitting
    scaled.freeze()
    assert scaled.is_frozen
    with pytest.raises(ValueError, match='frozen'):
        scaled.set_params(np.array([4.]))
##########################################################################################
