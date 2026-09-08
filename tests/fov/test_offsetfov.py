##########################################################################################
# tests/fov/test_offsetfov.py
##########################################################################################

import numpy as np
import pickle
import pytest

from polymath import Pair
from oops.fov import FlatFOV, OffsetFOV


def test_offsetfov():
    np.random.seed(4712)

    flat = FlatFOV((1.e-4, 1.2e-4), (60, 40))
    UV_OFFSET = (1., 2.)

    # An offset given in any array-like form becomes a Pair, so that the
    # Fittable interface, which reads uv_offset.vals, always works
    for arg in (UV_OFFSET, list(UV_OFFSET), np.array(UV_OFFSET),
                Pair(UV_OFFSET)):
        fov = OffsetFOV(flat, uv_offset=arg)
        assert isinstance(fov.uv_offset, Pair)
        assert isinstance(fov.xy_offset, Pair)
        assert fov.params == UV_OFFSET
        assert fov.nparams == 2

        # The fitting path accepts the same object
        fov.set_params((7., 8.))
        assert fov.params == (7., 8.)

    # The default is a zero offset
    assert OffsetFOV(flat).params == (0., 0.)

    # Only one of the two offsets may be given
    with pytest.raises(ValueError,
                       match='only one of uv_offset and xy_offset'):
        OffsetFOV(flat, UV_OFFSET, UV_OFFSET)

    # An offset given in (x,y) describes the same FOV as the equivalent
    # offset given in (u,v)
    by_uv = OffsetFOV(flat, uv_offset=UV_OFFSET)
    by_xy = OffsetFOV(flat, xy_offset=by_uv.xy_offset)
    assert by_xy.params == UV_OFFSET
    assert by_xy.uv_offset == by_uv.uv_offset

    # The offset displaces (x,y) by exactly xy_offset
    uv = Pair(np.random.rand(200, 2) * np.array([60., 40.]))
    assert by_uv.xy_from_uvt(uv) == flat.xy_from_uvt(uv) - by_uv.xy_offset

    # Pickling and copying both restore the offset. Only uv_offset is
    # saved, because the constructor derives xy_offset from it and refuses
    # to accept both.
    for fov in (by_uv, by_xy, OffsetFOV(flat)):
        for restored in (pickle.loads(pickle.dumps(fov)), fov.copy()):
            assert isinstance(restored, OffsetFOV)
            assert restored.params == fov.params
            assert restored.uv_offset == fov.uv_offset
            assert restored.xy_offset == fov.xy_offset
            assert restored.xy_from_uvt(uv) == fov.xy_from_uvt(uv)

    # A copy is fittable, and fitting it leaves the original alone
    original = OffsetFOV(flat, uv_offset=UV_OFFSET)
    copied = original.copy()
    assert copied is not original
    assert not copied.is_frozen
    copied.set_params((7., 8.))
    assert copied.params == (7., 8.)
    assert original.params == UV_OFFSET

    # The underlying FOV is shared rather than duplicated
    assert copied.fov is original.fov

    # An unpickled object, by contrast, is frozen
    assert pickle.loads(pickle.dumps(original)).is_frozen
##########################################################################################
