##########################################################################################
# tests/fov/test_gapfov.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath import Pair
from oops.fov import FlatFOV, GapFOV

FLAT = FlatFOV((1.e-4, 1.2e-4), (60, 40))


def test_gapfov_pixel_origins_are_unchanged() -> None:
    """Pixels have the same origins as in the given FOV, so integer (u,v) is unmoved."""

    fov = GapFOV(FLAT, 0.5)
    corners = Pair([(0., 0.), (1., 1.), (2., 7.), (30., 20.), (59., 39.)])

    assert fov.xy_from_uvt(corners) == FLAT.xy_from_uvt(corners)


def test_gapfov_pixel_extent_is_reduced() -> None:
    """Within a pixel, the offset from its origin is scaled by uv_size."""

    uv_size = 0.5
    fov = GapFOV(FLAT, uv_size)

    origin = Pair((3., 4.))
    offset = Pair((0.5, 0.5))

    # The point halfway across a gapped pixel sits uv_size as far from the pixel
    # origin as the same point in the underlying FOV
    xy_origin = FLAT.xy_from_uvt(origin)
    expected = xy_origin + (FLAT.xy_from_uvt(origin + offset) - xy_origin) * uv_size
    assert fov.xy_from_uvt(origin + offset).vals == pytest.approx(expected.vals)


def test_gapfov_uv_size_may_differ_by_axis() -> None:
    """A (u,v) pair of sizes shrinks the two axes independently."""

    fov = GapFOV(FLAT, (0.5, 0.25))

    origin = Pair((3., 4.))
    offset = Pair((1., 1.))
    xy_origin = FLAT.xy_from_uvt(origin)
    duv = FLAT.xy_from_uvt(origin + offset) - xy_origin

    xy = fov.xy_from_uvt(origin + Pair((0.5, 0.5)))
    assert xy.vals[0] == pytest.approx((xy_origin + duv * 0.5 * 0.5).vals[0])
    assert xy.vals[1] == pytest.approx((xy_origin + duv * 0.5 * 0.25).vals[1])


def test_gapfov_shape_is_inherited_but_scale_shrinks() -> None:
    """The pixel grid is unchanged; only the sensitive area of each pixel shrinks."""

    fov = GapFOV(FLAT, 0.5)

    assert fov.uv_shape == FLAT.uv_shape
    assert fov.uv_los == FLAT.uv_los
    assert fov.uv_scale == FLAT.uv_scale * 0.5
    assert fov.uv_area == pytest.approx(FLAT.uv_area * 0.25)


def test_gapfov_uv_size_accepts_any_arraylike() -> None:
    """A float, a tuple, and a Pair all describe the same GapFOV."""

    uv = Pair([(3.25, 4.75), (10.5, 30.5)])
    reference = GapFOV(FLAT, 0.5).xy_from_uvt(uv)

    for arg in (0.5, (0.5, 0.5), [0.5, 0.5], np.array([0.5, 0.5]), Pair((0.5, 0.5))):
        assert GapFOV(FLAT, arg).xy_from_uvt(uv) == reference


def test_gapfov_transform_is_invertible() -> None:
    """(u,v) -> (x,y) -> (u,v) returns the original coordinates."""

    np.random.seed(6221)
    fov = GapFOV(FLAT, 0.6)
    uv = Pair(np.random.rand(100, 2) * np.array([60., 40.]))

    assert abs(fov.uv_from_xyt(fov.xy_from_uvt(uv)) - uv).max() < 1.e-12
    assert fov.max_inversion_error() < 1.e-10


def test_gapfov_derivs_are_propagated() -> None:
    """With derivs=True, derivatives of (u,v) reach the returned (x,y)."""

    fov = GapFOV(FLAT, 0.5)
    uv = Pair([(3.25, 4.75), (10.5, 30.5)])
    uv.insert_deriv('t', Pair([(1., 1.), (1., 1.)]))

    assert 't' in fov.xy_from_uvt(uv, derivs=True).derivs
    assert 't' not in fov.xy_from_uvt(uv, derivs=False).derivs


def test_gapfov_remask_masks_points_outside() -> None:
    """remask=True masks (u,v) coordinates outside the field of view."""

    fov = GapFOV(FLAT, 0.5)
    uv = Pair([(30., 20.), (100., 20.), (30., 90.)])

    assert not np.any(fov.xy_from_uvt(uv, remask=False).mask)
    assert list(fov.xy_from_uvt(uv, remask=True).mask) == [False, True, True]


def test_gapfov_pickle() -> None:
    """Pickling restores both the underlying FOV and the pixel size."""

    fov = GapFOV(FLAT, (0.5, 0.25))
    uv = Pair([(3.25, 4.75), (10.5, 30.5)])
    restored = pickle.loads(pickle.dumps(fov))

    assert isinstance(restored, GapFOV)
    assert restored.uv_shape == fov.uv_shape
    assert restored.uv_scale == fov.uv_scale
    assert restored.uv_area == fov.uv_area
    assert restored.xy_from_uvt(uv) == fov.xy_from_uvt(uv)

##########################################################################################
