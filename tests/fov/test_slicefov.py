##########################################################################################
# tests/fov/test_slicefov.py
##########################################################################################

import pickle

import numpy as np

from polymath import Pair
from oops.fov import FlatFOV, SliceFOV

FLAT = FlatFOV((1.e-4, 1.2e-4), (60, 40))
ORIGIN = (10, 20)
SHAPE = (5, 8)


def test_slicefov_geometry_is_unchanged() -> None:
    """A slice is a window onto the reference FOV, so (u,v) is merely shifted."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)
    uv = Pair([(0., 0.), (1., 1.), (2.5, 3.5), (5., 8.)])

    assert fov.xy_from_uvt(uv) == FLAT.xy_from_uvt(uv + Pair(ORIGIN))


def test_slicefov_does_not_move_the_optic_axis() -> None:
    """This differs from a Subarray: the optic axis keeps its place in the sky, so in
    the slice's own coordinates it moves by the origin of the slice."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)

    assert fov.uv_los == FLAT.uv_los - Pair(ORIGIN)

    # The line of sight still points where it did in the reference FOV
    assert fov.xy_from_uvt(fov.uv_los) == Pair((0., 0.))


def test_slicefov_shape_and_scale() -> None:
    """The slice has the requested shape and the pixel geometry of the reference FOV."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)

    assert fov.uv_shape == Pair(SHAPE)
    assert fov.uv_scale == FLAT.uv_scale
    assert fov.uv_area == FLAT.uv_area


def test_slicefov_origin_and_shape_accept_any_arraylike() -> None:
    """A tuple, a list, an array, and a Pair all describe the same slice."""

    uv = Pair([(1.5, 2.5), (4., 7.)])
    reference = SliceFOV(FLAT, ORIGIN, SHAPE).xy_from_uvt(uv)

    for origin in (ORIGIN, list(ORIGIN), np.array(ORIGIN), Pair(ORIGIN)):
        for shape in (SHAPE, list(SHAPE), np.array(SHAPE), Pair(SHAPE)):
            assert SliceFOV(FLAT, origin, shape).xy_from_uvt(uv) == reference


def test_slicefov_scalar_shape_is_square() -> None:
    """A single number gives the slice the same extent along both axes."""

    assert SliceFOV(FLAT, ORIGIN, 4.).uv_shape == Pair((4., 4.))


def test_slicefov_transform_is_invertible() -> None:
    """(u,v) -> (x,y) -> (u,v) returns the original coordinates."""

    np.random.seed(9013)
    fov = SliceFOV(FLAT, ORIGIN, SHAPE)
    uv = Pair(np.random.rand(100, 2) * np.array(SHAPE, dtype='float'))

    assert abs(fov.uv_from_xyt(fov.xy_from_uvt(uv)) - uv).max() < 1.e-12
    assert fov.max_inversion_error() < 1.e-10


def test_slicefov_bounds_follow_the_slice() -> None:
    """Only the sliced window is inside; the rest of the reference FOV is outside."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)

    assert not fov.uv_is_outside(Pair((0., 0.)))
    assert not fov.uv_is_outside(Pair((2., 4.)))
    assert not fov.uv_is_outside(Pair((5., 8.)))          # inclusive by default
    assert fov.uv_is_outside(Pair((5., 8.)), inclusive=False)
    assert fov.uv_is_outside(Pair((6., 4.)))
    assert fov.uv_is_outside(Pair((2., -1.)))


def test_slicefov_derivs_are_propagated() -> None:
    """With derivs=True, derivatives of (u,v) reach the returned (x,y)."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)
    uv = Pair([(1.5, 2.5), (4., 7.)])
    uv.insert_deriv('t', Pair([(1., 1.), (1., 1.)]))

    assert 't' in fov.xy_from_uvt(uv, derivs=True).derivs
    assert 't' not in fov.xy_from_uvt(uv, derivs=False).derivs


def test_slicefov_remask_masks_points_outside() -> None:
    """remask=True masks (u,v) coordinates outside the field of view."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)
    uv = Pair([(2., 4.), (2., 30.), (60., 40.), (-20., -30.)])

    assert not np.any(fov.xy_from_uvt(uv, remask=False).mask)
    assert list(fov.xy_from_uvt(uv, remask=True).mask) == [False, True, True, True]


def test_slicefov_pickle() -> None:
    """Pickling restores the reference FOV, the origin, and the shape."""

    fov = SliceFOV(FLAT, ORIGIN, SHAPE)
    uv = Pair([(1.5, 2.5), (4., 7.)])
    restored = pickle.loads(pickle.dumps(fov))

    assert isinstance(restored, SliceFOV)
    assert restored.uv_shape == fov.uv_shape
    assert restored.uv_los == fov.uv_los
    assert restored.uv_scale == fov.uv_scale
    assert restored.xy_from_uvt(uv) == fov.xy_from_uvt(uv)

##########################################################################################
