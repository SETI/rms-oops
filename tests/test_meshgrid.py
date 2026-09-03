##########################################################################################
# tests/test_meshgrid.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath       import Pair, Scalar
from oops.constants import RPD
from oops.fov       import FlatFOV
from oops.meshgrid  import Meshgrid


@pytest.fixture
def fov() -> FlatFOV:
    """An 8x8 flat field of view with one-arcsecond pixels."""

    return FlatFOV((RPD/3600., RPD/3600.), (8, 8))


def test_meshgrid_constructors_accept_fov_kwargs(fov: FlatFOV) -> None:
    """Every entry point takes `fov_kwargs` and forwards it to the constructor."""

    assert Meshgrid(fov, (4, 4), fov_kwargs={}).shape == ()
    assert Meshgrid.for_fov(fov, fov_kwargs={}).shape == (8, 8)
    assert Meshgrid.for_fov_center(fov, fov_kwargs={}).shape == ()
    assert Meshgrid.for_shape(fov, (8, 8), 0, 1, fov_kwargs={}).shape == (8, 8)


def test_meshgrid_fov_kwargs_defaults_are_not_shared(fov: FlatFOV) -> None:
    """Omitting `fov_kwargs` gives each Meshgrid its own dictionary."""

    first = Meshgrid(fov, (1, 1))
    second = Meshgrid(fov, (2, 2))
    first._fov_kwargs['scale'] = 2.

    assert second._fov_kwargs == {}


def test_meshgrid_survives_a_pickle_round_trip(fov: FlatFOV) -> None:
    """__setstate__ restores the meshgrid through the renamed keyword."""

    grid = pickle.loads(pickle.dumps(Meshgrid.for_fov(fov)))

    assert grid.shape == (8, 8)
    assert grid.los().shape == (8, 8)


def test_meshgrid_for_fov_center_uses_the_given_origin(fov: FlatFOV) -> None:
    """`origin` selects the line of sight; omitting it centers on the FOV."""

    assert Meshgrid.for_fov_center(fov).uv == (4., 4.)
    assert Meshgrid.for_fov_center(fov, origin=(2., 3.)).uv == (2., 3.)


def test_meshgrid_rejects_an_oversample_below_one(fov: FlatFOV) -> None:
    """The oversample error names oversample, not undersample."""

    with pytest.raises(ValueError, match=r'invalid oversample: .*0\.'):
        Meshgrid.for_shape(fov, (8, 8), 0, 1, oversample=0)


def test_meshgrid_rejects_an_undersample_below_one(fov: FlatFOV) -> None:
    """An undersample below one is rejected by its own message."""

    with pytest.raises(ValueError, match=r'invalid undersample: .*0\.'):
        Meshgrid.for_shape(fov, (8, 8), 0, 1, undersample=0)


def test_meshgrid_rejects_sampling_in_both_directions(fov: FlatFOV) -> None:
    """Under- and oversampling cannot both be requested at once."""

    with pytest.raises(ValueError, match='cannot both be'):
        Meshgrid.for_shape(fov, (8, 8), 0, 1, undersample=2, oversample=2)


def test_meshgrid_with_no_u_axis_has_a_single_sample_along_u(fov: FlatFOV) -> None:
    """An axis absent from the shape contributes one sample, whatever the FOV spans.

    The shape reserves room for only one value along a missing axis, so sampling it as if
    it were present would leave more values than the result can hold.
    """

    meshgrid = Meshgrid.for_shape(fov, (8,), u_axis=-1, v_axis=0, limit=(8, 8))

    assert meshgrid.uv.shape == (8,)
    assert (meshgrid.uv.vals[..., 0] == 0.5).all()


def test_meshgrid_with_no_v_axis_has_a_single_sample_along_v(fov: FlatFOV) -> None:
    """The same holds for a missing v-axis."""

    meshgrid = Meshgrid.for_shape(fov, (8,), u_axis=0, v_axis=-1, limit=(8, 8))

    assert meshgrid.uv.shape == (8,)
    assert (meshgrid.uv.vals[..., 1] == 0.5).all()


def test_meshgrid_for_fov_samples_every_pixel(fov: FlatFOV) -> None:
    """By default the grid places one sample in the middle of each pixel."""

    grid = Meshgrid.for_fov(fov)

    assert grid.shape == (8, 8)
    assert grid.uv[0, 0] == Pair((0.5, 0.5))
    assert grid.uv[7, 7] == Pair((7.5, 7.5))


def test_meshgrid_for_fov_undersamples(fov: FlatFOV) -> None:
    """An undersample of 2 samples every other pixel along each axis."""

    grid = Meshgrid.for_fov(fov, undersample=2)

    assert grid.shape == (4, 4)


def test_meshgrid_for_fov_oversamples(fov: FlatFOV) -> None:
    """An oversample of 2 creates a 2x2 array of samples inside each pixel."""

    grid = Meshgrid.for_fov(fov, oversample=2)

    assert grid.shape == (16, 16)


def test_meshgrid_for_fov_honors_the_origin(fov: FlatFOV) -> None:
    """The origin places the first sample of the grid."""

    grid = Meshgrid.for_fov(fov, origin=(2., 3.))

    assert grid.uv[0, 0] == Pair((2., 3.))


def test_meshgrid_for_fov_honors_the_limit(fov: FlatFOV) -> None:
    """The limit is the upper bound of the grid, replacing the shape of the FOV."""

    grid = Meshgrid.for_fov(fov, limit=(4, 4))

    assert grid.shape == (4, 4)


def test_meshgrid_for_fov_can_swap_the_axes(fov: FlatFOV) -> None:
    """swap=True orders the indices (v,u) instead of (u,v)."""

    plain = Meshgrid.for_fov(fov, limit=(4, 8))
    swapped = Meshgrid.for_fov(fov, limit=(4, 8), swap=True)

    assert plain.shape == (4, 8)
    assert swapped.shape == (8, 4)


def test_meshgrid_for_fov_center_is_shapeless(fov: FlatFOV) -> None:
    """A center grid holds a single line of sight."""

    grid = Meshgrid.for_fov_center(fov)

    assert grid.shape == ()
    assert grid.uv == fov.uv_los


def test_meshgrid_center_uv_defaults_to_the_mean(fov: FlatFOV) -> None:
    """The center is the mean of the (u,v) values unless one is given."""

    grid = Meshgrid(fov, Pair([(0., 0.), (2., 4.)]))

    assert grid.center_uv == Pair((1., 2.))


def test_meshgrid_center_uv_may_be_given(fov: FlatFOV) -> None:
    """An explicit center replaces the mean of the samples."""

    grid = Meshgrid(fov, Pair([(0., 0.), (2., 4.)]), center_uv=Pair((5., 6.)))

    assert grid.center_uv == Pair((5., 6.))


def test_meshgrid_uv_carries_no_derivatives(fov: FlatFOV) -> None:
    """The plain (u,v) coordinates are free of derivatives."""

    grid = Meshgrid.for_fov(fov)

    assert grid.uv.derivs == {}


def test_meshgrid_uv_w_duv_duv_carries_the_identity(fov: FlatFOV) -> None:
    """The companion coordinates carry the identity derivative d(u,v)/d(u,v)."""

    grid = Meshgrid.for_fov(fov)

    assert 'uv' in grid.uv_w_duv_duv.derivs


def test_meshgrid_center_uv_w_duv_duv_carries_the_identity(fov: FlatFOV) -> None:
    """The center coordinates carry the identity derivative too."""

    grid = Meshgrid.for_fov(fov)

    assert 'uv' in grid.center_uv_w_duv_duv.derivs


def test_meshgrid_los_matches_the_fov(fov: FlatFOV) -> None:
    """The lines of sight are those the FOV gives for the same coordinates."""

    grid = Meshgrid.for_fov(fov)

    assert grid.los() == fov.los_from_uv(grid.uv)


def test_meshgrid_los_carries_no_derivatives(fov: FlatFOV) -> None:
    """The plain lines of sight are free of derivatives."""

    assert Meshgrid.for_fov(fov).los().derivs == {}


def test_meshgrid_los_w_derivs_carries_duv(fov: FlatFOV) -> None:
    """The companion lines of sight carry the derivative with respect to (u,v)."""

    assert 'uv' in Meshgrid.for_fov(fov).los_w_derivs().derivs


def test_meshgrid_dlos_duv_is_the_derivative_of_the_los(fov: FlatFOV) -> None:
    """dlos_duv is the derivative that los_w_derivs carries."""

    grid = Meshgrid.for_fov(fov)

    assert grid.dlos_duv() == grid.los_w_derivs().derivs['uv']


def test_meshgrid_duv_dlos_inverts_dlos_duv(fov: FlatFOV) -> None:
    """The two derivative matrices are inverses, so their product is the identity."""

    grid = Meshgrid.for_fov(fov)
    product = grid.duv_dlos()[0, 0].chain(grid.dlos_duv()[0, 0])

    assert product.vals == pytest.approx(np.identity(2), abs=1.e-9)


def test_meshgrid_uv_w_derivs_carries_dlos(fov: FlatFOV) -> None:
    """The (u,v) coordinates carry the derivative with respect to the line of sight."""

    assert 'los' in Meshgrid.for_fov(fov).uv_w_derivs().derivs


def test_meshgrid_center_los_is_the_los_at_the_center(fov: FlatFOV) -> None:
    """The central line of sight is the one the FOV gives for center_uv."""

    grid = Meshgrid.for_fov(fov)

    assert grid.center_los() == fov.los_from_uv(grid.center_uv)


def test_meshgrid_center_los_w_derivs_carries_duv(fov: FlatFOV) -> None:
    """The central line of sight carries the derivative with respect to (u,v)."""

    assert 'uv' in Meshgrid.for_fov(fov).center_los_w_derivs().derivs


def test_meshgrid_center_dlos_duv_is_the_derivative_of_the_center_los(
        fov: FlatFOV) -> None:
    """center_dlos_duv is the derivative that center_los_w_derivs carries."""

    grid = Meshgrid.for_fov(fov)

    assert grid.center_dlos_duv() == grid.center_los_w_derivs().derivs['uv']


def test_meshgrid_center_uv_w_derivs_carries_dlos(fov: FlatFOV) -> None:
    """The central (u,v) coordinates carry the derivative with respect to the los."""

    assert 'los' in Meshgrid.for_fov(fov).center_uv_w_derivs().derivs


def test_meshgrid_center_duv_dlos_is_the_derivative_of_the_center_uv(
        fov: FlatFOV) -> None:
    """center_duv_dlos is the derivative that center_uv_w_derivs carries."""

    grid = Meshgrid.for_fov(fov)

    assert grid.center_duv_dlos() == grid.center_uv_w_derivs().derivs['los']


@pytest.mark.parametrize('method', ['los', 'los_w_derivs', 'dlos_duv', 'uv_w_derivs',
                                    'duv_dlos', 'center_los', 'center_los_w_derivs',
                                    'center_dlos_duv', 'center_uv_w_derivs',
                                    'center_duv_dlos'])
def test_meshgrid_results_are_cached(method: str, fov: FlatFOV) -> None:
    """Every result is cached, so repeated calls at the same time are inexpensive."""

    grid = Meshgrid.for_fov(fov)

    assert getattr(grid, method)() is getattr(grid, method)()


@pytest.mark.parametrize('time', [None, 0., Scalar(100.)],
                         ids=['None', 'float', 'Scalar'])
def test_meshgrid_accepts_a_time(time, fov: FlatFOV) -> None:
    """A time-independent FOV gives the same lines of sight at any time."""

    grid = Meshgrid.for_fov(fov)

    assert grid.los(time) == grid.los()


def test_meshgrid_for_shape_broadcasts_to_the_given_shape(fov: FlatFOV) -> None:
    """The meshgrid is shaped so that it broadcasts to the observation's shape."""

    grid = Meshgrid.for_shape(fov, (8, 8), u_axis=0, v_axis=1)

    assert grid.shape == (8, 8)


def test_meshgrid_for_shape_places_the_axes(fov: FlatFOV) -> None:
    """The u and v axes take the positions named by u_axis and v_axis."""

    grid = Meshgrid.for_shape(fov, (8, 8), u_axis=1, v_axis=0)

    assert grid.uv[0, 0] == Pair((0.5, 0.5))
    assert grid.uv[0, 7] == Pair((7.5, 0.5))


def test_meshgrid_for_shape_honors_center_uv(fov: FlatFOV) -> None:
    """An explicit center replaces the center of the grid of points."""

    grid = Meshgrid.for_shape(fov, (8, 8), u_axis=0, v_axis=1, center_uv=(1., 2.))

    assert grid.center_uv == Pair((1., 2.))

##########################################################################################
