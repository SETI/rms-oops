##########################################################################################
# tests/fov/test_fov.py
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Vector3

from oops.fov       import FlatFOV, OffsetFOV, Platescale, TDIFOV
from oops.fov.fov_  import FOV


def _flat():
    """A time-independent FOV for these tests.

    Returns:
        FlatFOV: A 64 by 64 FOV with a pixel scale of 1e-4 radians.
    """

    return FlatFOV((1.e-4, 1.e-4), (64, 64))


def _tdi():
    """A time-dependent FOV for these tests.

    Returns:
        TDIFOV: A TDI FOV built on `_flat`, reading out along the v-axis.
    """

    return TDIFOV(_flat(), 100., 8., '-v')


CACHED_METHODS = ('center_xy', 'center_los', 'corner00_xy', 'corner01_xy', 'corner10_xy',
                  'corner11_xy')


@pytest.mark.parametrize('name', CACHED_METHODS)
def test_cached_value_is_reused_when_time_is_irrelevant(name) -> None:
    """A time-independent FOV caches each value and returns it for any time."""

    fov = _flat()
    method = getattr(fov, name)

    first = method()
    assert method() is first                    # the cached object, not a copy
    assert method(time=0.) is first             # time cannot matter here
    assert method(time=1.e8) is first


@pytest.mark.parametrize('name', CACHED_METHODS)
def test_a_time_dependent_fov_requires_a_time(name) -> None:
    """A time-dependent FOV cannot answer without a time, and says so."""

    fov = _tdi()

    with pytest.raises(NotImplementedError, match='time-dependent'):
        getattr(fov, name)()


@pytest.mark.parametrize('name', CACHED_METHODS)
def test_a_time_dependent_fov_caches_nothing(name) -> None:
    """A time-dependent FOV answers for the time it is given, whatever was asked before.

    Its values must not be cached, because a cached one would be returned for every later
    time as well.
    """

    fov = _tdi()
    method = getattr(fov, name)

    method(time=20.)
    method(time=60.)
    after_others = method(time=99.)

    assert after_others == getattr(_tdi(), name)(time=99.)
    assert not [key for key in FOV._CACHED_NAMES if key in fov.__dict__]


def test_refitting_an_offset_fov_discards_its_cached_values() -> None:
    """Moving an OffsetFOV moves the values cached from its former position."""

    fov = OffsetFOV(_flat(), uv_offset=(0., 0.))
    before = fov.center_xy()

    fov.set_params((10., 5.))

    assert fov.center_xy() != before


def test_refitting_a_platescale_discards_its_cached_values() -> None:
    """Rescaling a Platescale rescales the values cached from its former scale."""

    fov = Platescale(1., _flat())
    corner = fov.corner11_xy()
    radius = fov.outer_radius

    fov.set_params((2.,))

    # The refitted FOV reports what the same FOV would report had it been built this way.
    rebuilt = Platescale(2., _flat())
    assert fov.corner11_xy() == rebuilt.corner11_xy()
    assert fov.outer_radius == rebuilt.outer_radius

    assert fov.corner11_xy() != corner
    assert fov.outer_radius != radius


NEAREST_UV_INPUTS = ((70., 70.), [70., 70.], np.array([70., 70.]), Pair((70., 70.)))


@pytest.mark.parametrize('uv_pair', NEAREST_UV_INPUTS)
def test_nearest_uv_clips_any_accepted_input(uv_pair) -> None:
    """Every input form Pair.as_pair accepts is clipped to the FOV boundary."""

    assert _flat().nearest_uv(uv_pair) == Pair((64., 64.))


@pytest.mark.parametrize('uv_pair', NEAREST_UV_INPUTS)
def test_nearest_uv_remasks_any_accepted_input(uv_pair) -> None:
    """A point outside the FOV is masked with remask=True, whatever form it arrived in."""

    assert _flat().nearest_uv(uv_pair, remask=True).mask


def test_nearest_uv_leaves_a_point_inside_the_fov_alone() -> None:
    """A point already inside the FOV comes back unchanged."""

    assert _flat().nearest_uv((10., 20.)) == Pair((10., 20.))


def test_nearest_uv_does_not_mask_a_point_inside_the_fov() -> None:
    """remask=True masks only the points that had to move."""

    assert not _flat().nearest_uv((10., 20.), remask=True).mask


def test_nearest_uv_does_not_modify_its_argument() -> None:
    """The clip is applied to a copy; the caller's Pair is shared and must not move."""

    uv_pair = Pair([(10., 10.), (70., 70.)])
    _flat().nearest_uv(uv_pair)

    assert uv_pair == Pair([(10., 10.), (70., 70.)])


def test_nearest_uv_keeps_derivatives() -> None:
    """Derivatives survive the clip when the result is not remasked."""

    uv_pair = Pair([(10., 10.), (70., 70.)])
    uv_pair.insert_deriv('t', Pair([(1., 1.), (1., 1.)]))

    assert 't' in _flat().nearest_uv(uv_pair).derivs


##########################################################################################
# Radii, pixel offsets, and the nearest point inside the FOV
##########################################################################################

def test_outer_radius_circumscribes_the_field() -> None:
    """The outer radius reaches the corner of the field of view."""

    fov = _flat()
    corner = fov.los_from_uv(Pair((0., 0.)))
    center = fov.center_los()

    assert fov.outer_radius == pytest.approx(corner.sep(center).vals, rel=1.e-6)


def test_inner_radius_is_enclosed_by_the_field() -> None:
    """The inner radius reaches the nearest edge, so it is the smaller of the two."""

    fov = _flat()

    assert 0. < fov.inner_radius < fov.outer_radius


def test_inner_radius_reaches_the_edge() -> None:
    """A square FOV's inner radius reaches the midpoint of an edge."""

    fov = _flat()
    edge = fov.los_from_uv(Pair((0., 32.)))

    assert fov.inner_radius == pytest.approx(edge.sep(fov.center_los()).vals, rel=1.e-6)


def test_center_xy_is_the_optic_axis() -> None:
    """The center of the FOV is where (u,v) equals the line of sight."""

    assert _flat().center_xy() == Pair((0., 0.))


def test_center_los_points_along_z() -> None:
    """The center of a flat FOV looks straight down the Z-axis."""

    assert _flat().center_los() == Vector3((0., 0., 1.))


def test_center_dlos_duv_is_the_derivative_at_the_center() -> None:
    """The matrix carries a (u,v) denominator."""

    derivative = _flat().center_dlos_duv

    assert derivative.denom == (2,)


def test_a_time_dependent_fov_needs_a_time_for_its_center() -> None:
    """center_xy cannot be evaluated without a time when the FOV varies with it."""

    with pytest.raises(NotImplementedError):
        _tdi().center_xy()


def test_a_time_dependent_fov_needs_a_time_for_its_line_of_sight() -> None:
    """center_los cannot be evaluated without a time when the FOV varies with it."""

    with pytest.raises(NotImplementedError):
        _tdi().center_los()


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason='center_dlos_duv passes a time of None straight into the FOV '
                          'instead of raising NotImplementedError as its docstring and '
                          'its sibling properties do')
def test_a_time_dependent_fov_has_no_center_derivative() -> None:
    """The property takes no time at which to evaluate it."""

    with pytest.raises(NotImplementedError):
        _tdi().center_dlos_duv


def test_a_time_dependent_fov_has_no_outer_radius() -> None:
    """The property takes no time at which to evaluate it."""

    with pytest.raises(NotImplementedError):
        _tdi().outer_radius


def test_a_time_dependent_fov_has_no_inner_radius() -> None:
    """The property takes no time at which to evaluate it."""

    with pytest.raises(NotImplementedError):
        _tdi().inner_radius


def test_offset_angles_invert_the_pixel_offset() -> None:
    """The two conversions are inverses of one another."""

    fov = _flat()
    duv = Pair((1., 2.))

    restored = fov.offset_duv_from_angles(fov.offset_angles_from_duv(duv))

    assert restored.vals == pytest.approx(duv.vals, abs=1.e-9)


def test_offset_angles_scale_with_the_pixel_size() -> None:
    """A one-pixel offset rotates by about one pixel's worth of angle."""

    fov = _flat()
    (about_y, about_x) = fov.offset_angles_from_duv(Pair((1., 0.)))

    assert abs(about_y.vals) == pytest.approx(1.e-4, rel=1.e-3)
    assert about_x.vals == pytest.approx(0., abs=1.e-12)


def test_a_zero_offset_gives_zero_angles() -> None:
    """No displacement means no rotation."""

    (about_y, about_x) = _flat().offset_angles_from_duv(Pair((0., 0.)))

    assert about_y.vals == pytest.approx(0., abs=1.e-15)
    assert about_x.vals == pytest.approx(0., abs=1.e-15)


def test_offset_angles_depend_on_the_origin() -> None:
    """The angles are measured at a reference location, which defaults to the center."""

    fov = _flat()
    duv = Pair((1., 2.))

    at_center = fov.offset_angles_from_duv(duv)
    off_center = fov.offset_angles_from_duv(duv, origin=Pair((5., 5.)))

    assert at_center[0].vals != off_center[0].vals


def test_nearest_uv_leaves_an_interior_point_alone() -> None:
    """A point already inside the FOV is its own nearest point."""

    assert _flat().nearest_uv(Pair((30., 20.))) == Pair((30., 20.))


def test_nearest_uv_pulls_an_exterior_point_to_the_edge() -> None:
    """A point outside is moved to the closest point on the boundary."""

    assert _flat().nearest_uv(Pair((100., 20.))) == Pair((64., 20.))


def test_nearest_uv_clamps_both_axes() -> None:
    """A point beyond a corner is pulled back to that corner."""

    assert _flat().nearest_uv(Pair((-10., 100.))) == Pair((0., 64.))


def test_nearest_uv_can_mask_the_exterior_points() -> None:
    """remask=True masks the points that fell outside the boundary."""

    nearest = _flat().nearest_uv(Pair([(100., 20.), (30., 20.)]), remask=True)

    assert list(nearest.mask) == [True, False]


def test_nearest_uv_accepts_any_arraylike() -> None:
    """A tuple, a list, and an array all describe the same coordinates."""

    fov = _flat()
    expected = fov.nearest_uv(Pair((30., 20.)))

    for arg in ((30., 20.), [30., 20.], np.array([30., 20.])):
        assert fov.nearest_uv(arg) == expected

##########################################################################################
# uv_from_los, the outside tests, and the one-axis variant
##########################################################################################

def test_uv_from_los_works_on_a_time_independent_fov() -> None:
    """Without time dependence, the line of sight alone determines the pixel."""

    fov = _flat()
    uv = Pair([(10.5, 20.5), (30.25, 40.75)])

    assert fov.uv_from_los(fov.los_from_uvt(uv)) == uv


def test_uv_from_los_is_refused_on_a_time_dependent_fov() -> None:
    """A TDI FOV needs a time, so the timeless entry point is not available."""

    with pytest.raises(NotImplementedError, match='TDIFOV.uv_from_los'):
        _tdi().uv_from_los(Vector3((0., 0., 1.)))


def test_uv_is_outside_marks_the_points_beyond_the_field() -> None:
    """A point outside either axis of the field of view is outside it."""

    outside = _flat().uv_is_outside(Pair([(10., 20.), (100., 20.), (10., 90.)]))

    assert list(outside.vals) == [False, True, True]


def test_uv_is_outside_can_be_restricted_to_a_sub_rectangle() -> None:
    """Explicit corners replace the full extent of the field of view."""

    fov = _flat()
    uv = Pair([(10., 20.), (40., 20.)])

    assert list(fov.uv_is_outside(uv, uv_min=Pair((0, 0)),
                                 uv_max=Pair((32, 32))).vals) == [False, True]


def test_u_or_v_is_outside_tests_one_axis_alone() -> None:
    """Only the coordinate named by uv_index is tested."""

    fov = _flat()
    uv = Pair([(10., 90.), (100., 20.)])

    assert list(fov.u_or_v_is_outside(uv, 0).vals) == [False, True]
    assert list(fov.u_or_v_is_outside(uv, 1).vals) == [True, False]


def test_u_or_v_is_outside_can_be_restricted_to_a_sub_range() -> None:
    """Explicit corners replace the full extent along the selected axis."""

    fov = _flat()
    uv = Pair([(10., 20.), (40., 20.)])

    assert list(fov.u_or_v_is_outside(uv, 0, uv_min=Pair((0, 0)),
                                      uv_max=Pair((32, 32))).vals) == [False, True]


def test_u_or_v_is_outside_can_exclude_the_upper_edge() -> None:
    """inclusive=False treats the far edge of the range as outside."""

    fov = _flat()
    uv = Pair((64., 32.))

    assert not fov.u_or_v_is_outside(uv, 0)
    assert fov.u_or_v_is_outside(uv, 0, inclusive=False)


def test_xy_is_outside_maps_the_coordinates_back_to_the_field() -> None:
    """The (x,y) coordinates are converted to (u,v) before the test is applied."""

    fov = _flat()
    uv = Pair([(10., 20.), (100., 20.)])

    assert list(fov.xy_is_outside(fov.xy_from_uvt(uv)).vals) == [False, True]


def test_los_is_outside_maps_the_direction_back_to_the_field() -> None:
    """A line of sight is converted to (x,y) and then to (u,v) before the test."""

    fov = _flat()
    uv = Pair([(10., 20.), (100., 20.)])

    assert list(fov.los_is_outside(fov.los_from_uvt(uv)).vals) == [False, True]

##########################################################################################
# sphere_falls_inside
##########################################################################################

# A sphere at this distance along the FOV axis subtends about 0.001 radians, which is
# roughly a third of the half-width of the 64x64 flat FOV
SPHERE_RANGE = 1.e6
SPHERE_RADIUS = 1.e3


def _sphere_center(uv) -> Vector3:
    """The center of a test sphere, seen through a pixel of the flat FOV.

    Parameters:
        uv: The (u,v) coordinates of the pixel it is seen through.

    Returns:
        Vector3: The position of the sphere's center.
    """

    return _flat().los_from_uvt(Pair.as_pair(uv)).unit() * SPHERE_RANGE


def test_a_sphere_at_the_center_falls_inside() -> None:
    """A sphere on the optic axis is well inside the field of view."""

    assert _flat().sphere_falls_inside(_sphere_center((32., 32.)), SPHERE_RADIUS)


def test_a_sphere_far_off_the_axis_falls_outside() -> None:
    """A sphere many fields away is nowhere near the field of view."""

    assert not _flat().sphere_falls_inside(_sphere_center((1000., 1000.)),
                                           SPHERE_RADIUS)


def test_a_sphere_just_beyond_the_corner_can_still_reach_inside() -> None:
    """A sphere centered outside the field still falls inside if its limb reaches in."""

    fov = _flat()
    just_outside = _sphere_center((64.5, 64.5))

    assert not fov.sphere_falls_inside(just_outside, 1.)
    assert fov.sphere_falls_inside(just_outside, 1.e4)


def test_a_border_extends_the_field_the_sphere_is_tested_against() -> None:
    """The border is added to the field of view before the test is applied."""

    fov = _flat()
    center = _sphere_center((70., 32.))

    assert not fov.sphere_falls_inside(center, 1.)
    assert fov.sphere_falls_inside(center, 1., border=0.001)

##########################################################################################
