##########################################################################################
# tests/fov/test_nullfov.py
##########################################################################################

import pickle

from polymath import Boolean, Pair, Scalar, Vector3
from oops.fov import NullFOV


def test_nullfov_geometry() -> None:
    """A NullFOV describes an instrument with no field of view."""

    fov = NullFOV()

    assert fov.uv_shape == Pair((1, 1))
    assert fov.uv_los == Pair((0., 0.))
    assert fov.uv_scale == Pair((1., 1.))
    assert fov.uv_area == 1.


def test_nullfov_xy_is_always_the_optic_axis() -> None:
    """Every (u,v) maps to the same (x,y), because the FOV has no extent."""

    fov = NullFOV()

    for uv in [(0., 0.), (0.5, 0.5), (1., 1.), (-30., 44.)]:
        assert fov.xy_from_uvt(Pair(uv)) == Pair((0., 0.))


def test_nullfov_uv_is_always_the_optic_axis() -> None:
    """The reverse transform likewise collapses onto the FOV's single (u,v)."""

    fov = NullFOV()

    for xy in [(0., 0.), (1.e-4, -2.e-4), (3., 4.)]:
        assert fov.uv_from_xyt(Pair(xy)) == Pair((0., 0.))


def test_nullfov_area_factor_is_unity() -> None:
    """The relative area is scaled to the nominal pixel area, so it is one."""

    fov = NullFOV()

    for uv in [(0., 0.), (0.5, 0.5), (10., -10.)]:
        assert fov.area_factor(Pair(uv)) == Scalar(1.)


def test_nullfov_line_of_sight_is_the_z_axis() -> None:
    """A NullFOV looks along +Z, the direction opposite the arriving photon."""

    fov = NullFOV()

    assert fov.los_from_xy(Pair((0., 0.))) == Vector3((0., 0., 1.))
    assert fov.los_from_uvt(Pair((7., -3.))) == Vector3((0., 0., 1.))
    assert fov.center_los() == Vector3((0., 0., 1.))
    assert fov.center_xy() == Pair((0., 0.))


def test_nullfov_xy_from_los_is_the_optic_axis() -> None:
    """Every line of sight maps to the one (x,y) this FOV has."""

    fov = NullFOV()

    for los in [(0., 0., 1.), (0., 0., 5.), (0.3, -0.4, 2.)]:
        assert fov.xy_from_los(Vector3(los)) == Pair((0., 0.))


def test_nullfov_uv_from_los_t_collapses() -> None:
    """Every line of sight maps back to the FOV's single (u,v)."""

    fov = NullFOV()

    for los in [(0., 0., 1.), (0.1, 0.2, 1.)]:
        assert fov.uv_from_los_t(Vector3(los)) == Pair((0., 0.))


def test_nullfov_nearest_uv_is_the_optic_axis() -> None:
    """The only (u,v) this FOV has is (0,0), so that is the nearest point."""

    fov = NullFOV()

    for uv in [(0., 0.), (0.5, 0.5), (100., -100.)]:
        assert fov.nearest_uv(Pair(uv)) == Pair((0., 0.))


def test_nullfov_everything_is_outside() -> None:
    """A NullFOV has no field of view, so no coordinate falls inside it."""

    fov = NullFOV()

    for uv in [(0., 0.), (0.5, 0.5), (1., 1.), (5., 5.), (-1., -1.)]:
        assert fov.uv_is_outside(Pair(uv)) == Boolean(True)
        assert fov.uv_is_outside(Pair(uv), inclusive=False) == Boolean(True)
        assert fov.u_or_v_is_outside(Pair(uv), 0) == Boolean(True)
        assert fov.u_or_v_is_outside(Pair(uv), 1) == Boolean(True)

    assert fov.xy_is_outside(Pair((0., 0.))) == Boolean(True)
    assert fov.los_is_outside(Vector3((0., 0., 1.))) == Boolean(True)


def test_nullfov_max_inversion_error_is_zero() -> None:
    """A NullFOV is not invertible, so zero is returned rather than the sampled extent."""

    fov = NullFOV()

    assert fov.max_inversion_error() == 0.
    assert fov.max_inversion_error(steps=5) == 0.


def test_nullfov_pickle() -> None:
    """A NullFOV carries no state, so a restored copy behaves identically."""

    fov = NullFOV()
    restored = pickle.loads(pickle.dumps(fov))

    assert isinstance(restored, NullFOV)
    assert restored.uv_shape == fov.uv_shape
    assert restored.uv_los == fov.uv_los
    assert restored.uv_scale == fov.uv_scale
    assert restored.uv_area == fov.uv_area
    assert restored.xy_from_uvt(Pair((2., 3.))) == fov.xy_from_uvt(Pair((2., 3.)))

##########################################################################################
