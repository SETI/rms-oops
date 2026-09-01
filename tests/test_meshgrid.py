##########################################################################################
# tests/test_meshgrid.py
##########################################################################################

import pickle

import pytest

import oops
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

##########################################################################################
