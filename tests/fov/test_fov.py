##########################################################################################
# tests/fov/test_fov.py
##########################################################################################

import pytest

from oops.fov import FlatFOV, OffsetFOV, Platescale, TDIFOV


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
    assert not [key for key in fov.__dict__ if key.endswith('_filled')]


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

##########################################################################################
